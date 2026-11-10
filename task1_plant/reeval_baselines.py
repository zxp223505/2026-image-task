# -*- coding: utf-8 -*-
"""
回评 v1 / v2 已存下来的 checkpoint —— 用干净验证集，不重新训练

背景：v1/v2 训练时验证集误用了训练增强（Subset 不换 transform），
所以它们控制台打印的 val_f1 是在随机裁剪/翻转/颜色抖动后的图上算的，
不可信，也没留下记录。这里用 eval_tf（确定性变换）重新评估，
得到横向上可比的数字。

配对规则：checkpoint 的节号 k 对应同一折划分（random_state=42）的第 k 折验证集。
分辨率必须用各自训练时的：v1=224，v2=288，否则属于分布外评估。
架构从 state_dict 自动推断（fc.weight 的输入维度：512->ResNet18，2048->ResNet50）。

用法：python reeval_baselines.py
"""
import os
import re
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision import models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
T1 = r"C:\Users\zhaopangpang\Desktop\2026task1"
DATA_ROOT = os.path.join(T1, "dataset-for-task2", "dataset-for-task2")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 42
KFOLD = 5

# 待回评的来源：目录 + 训练时的输入分辨率
SOURCES = [
    {"label": "v1 ResNet18@224", "dir": T1, "img": 224},
    {"label": "v2 ResNet50@288", "dir": os.path.join(T1, "v2_resnet50"), "img": 288},
]


def to_rgb(img):
    return img.convert("RGB")


def eval_tf_for(img_size):
    return T.Compose([
        T.Lambda(to_rgb),
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def infer_arch(sd):
    """从 state_dict 里 fc.weight 的输入维度推断骨干"""
    if "fc.weight" in sd:
        fan_in = sd["fc.weight"].shape[1]
        return {512: "resnet18", 2048: "resnet50", 1024: "resnet34"}.get(fan_in)
    if "classifier.1.weight" in sd:
        return "efficientnet_b0"
    return None


def build_model(arch, n_classes):
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == "resnet34":
        m = models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    else:
        raise ValueError(arch)
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append(out.argmax(1).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}\n", flush=True)

    # 折划分：v1/v2 训练时都是 KFOLD=5, shuffle=True, random_state=42
    # 类别按文件名排序，ImageFolder 与它们一致
    probe = ImageFolder(TRAIN_DIR)
    classes = probe.classes
    n_classes = len(classes)
    y_all = np.array([s[1] for s in probe.samples])
    skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y_all)), y_all))
    print(f"类别({n_classes}): {classes}")
    print(f"折划分: StratifiedKFold(n_splits={KFOLD}, shuffle=True, random_state={SEED})")
    print("每折验证集大小:", [len(va) for _, va in folds], "\n", flush=True)

    all_results = {}
    for src in SOURCES:
        ckpts = sorted(glob.glob(os.path.join(src["dir"], "fold*_best.pt")))
        if not ckpts:
            print(f"[skip] {src['label']}: 目录下没有 fold*_best.pt")
            continue

        # 该分辨率下的验证集（配对用），建一次复用
        ds_eval = ImageFolder(TRAIN_DIR, transform=eval_tf_for(src["img"]))

        print(f"=== {src['label']} | {len(ckpts)} 个 checkpoint ===")
        f1s, accs, per_fold = [], [], []
        for ckpt in ckpts:
            m_ = re.search(r"fold(\d+)_best\.pt$", os.path.basename(ckpt))
            if not m_:
                continue
            fold = int(m_.group(1))
            if fold >= KFOLD:
                print(f"  [skip] {os.path.basename(ckpt)}: fold {fold} 超出 {KFOLD} 折划分")
                continue

            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
            arch = infer_arch(sd)
            model = build_model(arch, n_classes)
            model.load_state_dict(sd)
            model = model.to(device)

            _, va_idx = folds[fold]
            loader = DataLoader(Subset(ds_eval, va_idx), batch_size=16,
                                shuffle=False, num_workers=4)
            trues, preds = evaluate(model, loader, device)
            f1 = f1_score(trues, preds, average="macro")
            acc = accuracy_score(trues, preds)
            f1s.append(f1)
            accs.append(acc)
            per_fold.append({"fold": fold, "arch": arch, "f1": f1, "acc": acc})

            print(f"  fold{fold}  {arch:9s}  macro-F1={f1:.4f}  acc={acc:.4f}"
                  f"  (验证集 {len(va_idx)} 张)", flush=True)
            del model
            torch.cuda.empty_cache()

        if f1s:
            mean_f1, std_f1 = float(np.mean(f1s)), float(np.std(f1s))
            print(f"  --> {src['label']} 平均 macro-F1 = {mean_f1:.4f} ± {std_f1:.4f}"
                  f"   (基于 {len(f1s)} 折)", flush=True)

            # 最后一个 checkpoint 的逐类报告，用来看哪两类容易混
            print(f"  逐类 F1（fold{per_fold[-1]['fold']}）:")
            report = classification_report(trues, preds, target_names=classes,
                                           digits=4, zero_division=0)
            for line in report.splitlines():
                print("    " + line)
            print(flush=True)

            all_results[src["label"]] = {
                "per_fold": per_fold, "mean_f1": mean_f1, "std_f1": std_f1,
                "n_folds_evaluated": len(f1s),
            }

    out = os.path.join(OUT_DIR, "baseline-reeval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"结果已写入: {out}", flush=True)

    # 汇总对比
    print("\n===== 汇总（干净验证集，同一折划分）=====")
    for k, v in all_results.items():
        print(f"  {k:20s} macro-F1 = {v['mean_f1']:.4f} ± {v['std_f1']:.4f}"
              f"  ({v['n_folds_evaluated']} 折)")
    if len(all_results) == 2:
        ks = list(all_results)
        d = all_results[ks[1]]["mean_f1"] - all_results[ks[0]]["mean_f1"]
        print(f"\n  差值 ({ks[1]} - {ks[0]}) = {d:+.4f}")


if __name__ == "__main__":
    main()
