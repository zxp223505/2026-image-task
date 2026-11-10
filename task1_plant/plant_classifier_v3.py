# -*- coding: utf-8 -*-
"""
植物幼苗分类 v3 —— 在 v2 基础上修 bug，模型结构保持不变

相比 v2 只改了这三处（架构、超参、数据增强策略都原样保留）：
  1. 验证集不再误用训练增强
     v1/v2 里 full_ds = ImageFolder(TRAIN_DIR, transform=train_tf)，
     然后用 Subset(full_ds, va_idx) 做验证集 —— 但 Subset 不会换 transform，
     所以验证集也在做 RandomResizedCrop/翻转/旋转/ColorJitter，
     val_f1 每次都不一样，"best checkpoint" 是照着一个噪声指标选的。
     这里改为建两个 ImageFolder，训练用 train_tf、验证用 eval_tf。
     （ImageFolder 按类名排序，两个实例的 class_to_idx 必然一致）
  2. 提交文件行顺序对齐官方样例 submission-for-task2.csv
     官方样例不是字典序。评分若按行位置比对，顺序错了分会全崩。
  3. checkpoint / 日志 / 提交文件统一落到本脚本所在目录，不再依赖 cwd

用法：
    python plant_classifier_v3.py                    # ResNet50，5 折，带 TTA
    python plant_classifier_v3.py --arch resnet18    # 换回 ResNet18
    python plant_classifier_v3.py --no-tta           # 关掉 TTA
"""
import os
import csv
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision import models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image

# ---------------- 路径 ----------------
# 脚本所在目录 = 新文件夹，所有产物都写到这里
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = r"C:\Users\zhaopangpang\Desktop\2026task1\dataset-for-task2\dataset-for-task2"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
# 官方样例提交，用来对齐行顺序
SAMPLE_SUB = r"C:\Users\zhaopangpang\Desktop\submission-for-task2.csv"

# ---------------- 超参（与 v2 一致）----------------
IMG_SIZE = 288
BATCH = 16
EPOCHS = 25
KFOLD = 5
NUM_WORKERS = 4
MIXUP_ALPHA = 0.2
LABEL_SMOOTHING = 0.1

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def to_rgb(img):
    """统一转 RGB，处理偶发的 RGBA 图片（训练集里约 3%）"""
    return img.convert("RGB")


def build_transforms(img_size):
    """按输入分辨率构建 训练 / 评估 / TTA 三组变换。

    训练组带随机增强，评估组必须是确定性的 —— 两者分开是 v3 的核心修复，
    v1/v2 因为共用同一个 ImageFolder 的 transform，验证集也被加了随机增强。
    """
    train_tf = T.Compose([
        T.Lambda(to_rgb),
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_tf = T.Compose([
        T.Lambda(to_rgb),
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # TTA：原图 + 水平翻转
    tta_tfs = [
        eval_tf,
        T.Compose([T.Lambda(to_rgb), T.Resize(int(img_size * 1.14)),
                   T.CenterCrop(img_size), T.RandomHorizontalFlip(p=1.0),
                   T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    ]
    return train_tf, eval_tf, tta_tfs


class TestDataset(Dataset):
    """测试集是扁平结构（无类别子文件夹），按给定路径列表加载"""

    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, 0  # 占位标签


def build_model(n_classes, arch="resnet50"):
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif arch == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    else:
        raise ValueError(arch)
    return model


def mixup_data(x, y, alpha):
    """返回 (mixed_x, y_a, y_b, lam)"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append(out.argmax(1).cpu().numpy())
        trues.append(y.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return (f1_score(trues, preds, average="macro"),
            accuracy_score(trues, preds))


def train_one_fold(model, train_loader, val_loader, epochs, device, fold):
    model = model.to(device)
    head_params = list(model.fc.parameters()) if hasattr(model, "fc") \
        else list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters()
                       if not n.startswith(("fc.", "classifier."))]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 8e-5},
        {"params": head_params, "lr": 8e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    ckpt_path = os.path.join(OUT_DIR, f"fold{fold}_best.pt")
    best_f1 = 0.0
    history = []
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x, y_a, y_b, lam = mixup_data(x, y, MIXUP_ALPHA)
            optimizer.zero_grad()
            out = model(x)
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 验证集现在用的是 eval_tf，指标是干净可比的
        val_f1, val_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "val_f1": val_f1, "val_acc": val_acc})
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  [fold{fold}] epoch {epoch+1}/{epochs}  "
                  f"val_f1={val_f1:.4f}  val_acc={val_acc:.4f}", flush=True)
    print(f"  [fold{fold}] best val macro-F1 = {best_f1:.4f}", flush=True)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return best_f1, history


@torch.no_grad()
def predict_proba(model, paths, device, eval_tf, tta_tfs, tta=False):
    """返回 softmax 概率 [N, C]；tta=True 时对每个 TTA 变换求平均"""
    model.eval()
    transforms_list = tta_tfs if tta else [eval_tf]
    agg = None
    for tf in transforms_list:
        ds = TestDataset(paths, transform=tf)
        dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
        probs = []
        for x, _ in dl:
            out = model(x.to(device))
            probs.append(torch.softmax(out, dim=1).cpu().numpy())
        p = np.concatenate(probs)
        agg = p if agg is None else agg + p
    return agg / len(transforms_list)


def load_canonical_order(test_ids):
    """读官方样例提交，取它的 ID 行顺序。

    返回和 test_ids 等长的 ID 列表。若样例缺失或 ID 集合对不上，
    退回字典序并给出警告。
    """
    if not os.path.exists(SAMPLE_SUB):
        print(f"[warn] 找不到样例提交 {SAMPLE_SUB}，退回字典序")
        return test_ids
    with open(SAMPLE_SUB, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    canon = [r[0] for r in rows[1:] if r]
    if set(canon) != set(test_ids) or len(canon) != len(test_ids):
        print(f"[warn] 样例提交的 ID 与 test 目录不一致"
              f"（样例 {len(canon)} 个 / test {len(test_ids)} 个），退回字典序")
        return test_ids
    print(f"[ok] 提交行顺序已对齐官方样例（{'字典序' if canon == test_ids else '与字典序不同'}）")
    return canon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE,
                        help="输入分辨率（v1 用 224，v2 用 288）")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--kfold", type=int, default=KFOLD)
    parser.add_argument("--no-tta", dest="tta", action="store_false", default=True,
                        help="关闭 TTA（默认开启）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  arch={args.arch}  img={args.img_size}  "
          f"epochs={args.epochs}  kfold={args.kfold}  tta={args.tta}", flush=True)

    train_tf, eval_tf, tta_tfs = build_transforms(args.img_size)

    # --- 关键修复 1：训练/验证用各自独立的 ImageFolder ---
    ds_train = ImageFolder(TRAIN_DIR, transform=train_tf)
    ds_eval = ImageFolder(TRAIN_DIR, transform=eval_tf)
    assert ds_train.class_to_idx == ds_eval.class_to_idx, "两次扫描的类别映射不一致"
    classes = ds_train.classes
    n_classes = len(classes)
    idx_to_class = {v: k for k, v in ds_train.class_to_idx.items()}
    print(f"类别({n_classes}): {classes}", flush=True)

    test_ids_sorted = sorted(os.listdir(TEST_DIR))
    test_paths = [os.path.join(TEST_DIR, f) for f in test_ids_sorted]
    canon_order = load_canonical_order(test_ids_sorted)

    # 样例顺序 -> 在 sorted 列表里的下标
    pos_in_sorted = {fid: i for i, fid in enumerate(test_ids_sorted)}
    order_idx = [pos_in_sorted[fid] for fid in canon_order]

    y_all = np.array([s[1] for s in ds_train.samples])
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)

    fold_f1s = []
    fold_history = {}
    test_proba_sum = None
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all)):
        # 同一组下标，分别索引进 train 增强 / eval 变换的两个数据集
        tr_loader = DataLoader(Subset(ds_train, tr_idx), batch_size=BATCH,
                               shuffle=True, num_workers=NUM_WORKERS)
        va_loader = DataLoader(Subset(ds_eval, va_idx), batch_size=BATCH,
                               shuffle=False, num_workers=NUM_WORKERS)
        model = build_model(n_classes, args.arch)
        f1, hist = train_one_fold(model, tr_loader, va_loader,
                                  args.epochs, device, fold)
        fold_f1s.append(f1)
        fold_history[f"fold{fold}"] = hist
        p = predict_proba(model, test_paths, device, eval_tf, tta_tfs, tta=args.tta)
        test_proba_sum = p if test_proba_sum is None else test_proba_sum + p
        del model
        torch.cuda.empty_cache()

    mean_f1, std_f1 = float(np.mean(fold_f1s)), float(np.std(fold_f1s))
    print(f"\n==> {args.kfold} 折交叉验证 macro-F1: {mean_f1:.4f} ± {std_f1:.4f}",
          flush=True)

    # 各 fold 概率平均后取 argmax
    final_proba = test_proba_sum / args.kfold
    pred_idx = final_proba.argmax(1)
    pred_labels = [idx_to_class[i] for i in pred_idx]

    # --- 关键修复 2：按官方样例的行顺序写出 ---
    tag = f"{args.arch}_{args.img_size}"
    out_csv = os.path.join(OUT_DIR, f"submission-v3-{tag}.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("ID,Category\n")
        for pos in order_idx:
            f.write(f"{test_ids_sorted[pos]},{pred_labels[pos]}\n")

    print(f"submission 已写入: {out_csv}  ({len(canon_order)} 行)", flush=True)

    from collections import Counter
    dist = dict(Counter(pred_labels))
    print("预测类别分布:", dist, flush=True)

    # 存一份运行记录，方便之后对比不同架构
    log = {
        "arch": args.arch, "img_size": args.img_size, "epochs": args.epochs,
        "kfold": args.kfold, "tta": args.tta, "mixup_alpha": MIXUP_ALPHA,
        "label_smoothing": LABEL_SMOOTHING,
        "fold_f1": fold_f1s, "mean_f1": mean_f1, "std_f1": std_f1,
        "pred_dist": dist, "history": fold_history,
    }
    log_path = os.path.join(OUT_DIR, f"run-log-{tag}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"运行记录已写入: {log_path}", flush=True)


if __name__ == "__main__":
    main()
