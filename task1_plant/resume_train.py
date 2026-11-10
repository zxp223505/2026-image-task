# -*- coding: utf-8 -*-
"""
续训脚本 —— 从指定折继续训练，避免每次从头重跑 fold0。

背景：完整训练约 2.5 小时，后台任务反复在 fold1 附近被回收。fold0 的
best 权重已经保存，没必要每次重训。本脚本：

  1. 复用 plant_classifier_v3.py 里的一切（模型、数据、训练、TTA、行序对齐）
  2. 对 start_fold 之前的折：加载已保存的 fold{N}_best.pt，直接在验证集回评
     F1 并对测试集做 TTA 预测（不重新训练）
  3. 对 start_fold 及之后的折：正常训练
  4. 汇总 5 折概率 -> 写 submission CSV 和 run-log

用法：
    python resume_train.py --arch resnet18 --img-size 224 --start-fold 1
"""
import os
import sys
import csv
import json
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold

# 让本目录下的 plant_classifier_v3 可以被 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plant_classifier_v3 as v3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--start-fold", type=int, default=1,
                        help="从第几折开始训练（0=从头；之前的折用已存权重复用）")
    parser.add_argument("--epochs", type=int, default=v3.EPOCHS)
    parser.add_argument("--kfold", type=int, default=v3.KFOLD)
    parser.add_argument("--no-tta", dest="tta", action="store_false", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"resume: device={device} arch={args.arch} img={args.img_size} "
          f"start_fold={args.start_fold} epochs={args.epochs} kfold={args.kfold} "
          f"tta={args.tta}", flush=True)

    # 校验 start_fold 之前的权重都在，否则无从续起
    for fold in range(args.start_fold):
        ckpt = os.path.join(v3.OUT_DIR, f"fold{fold}_best.pt")
        if not os.path.exists(ckpt):
            raise SystemExit(f"[error] 缺少 {ckpt}，无法从 fold{args.start_fold} 续训")

    train_tf, eval_tf, tta_tfs = v3.build_transforms(args.img_size)
    ds_train = ImageFolder(v3.TRAIN_DIR, transform=train_tf)
    ds_eval = ImageFolder(v3.TRAIN_DIR, transform=eval_tf)
    assert ds_train.class_to_idx == ds_eval.class_to_idx
    classes = ds_train.classes
    n_classes = len(classes)
    idx_to_class = {v: k for k, v in ds_train.class_to_idx.items()}
    print(f"类别({n_classes}): {classes}", flush=True)

    test_ids_sorted = sorted(os.listdir(v3.TEST_DIR))
    test_paths = [os.path.join(v3.TEST_DIR, f) for f in test_ids_sorted]
    canon_order = v3.load_canonical_order(test_ids_sorted)
    pos_in_sorted = {fid: i for i, fid in enumerate(test_ids_sorted)}
    order_idx = [pos_in_sorted[fid] for fid in canon_order]

    y_all = np.array([s[1] for s in ds_train.samples])
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(len(y_all)), y_all))

    fold_f1s = []
    fold_history = {}
    test_proba_sum = None

    for fold, (tr_idx, va_idx) in enumerate(folds):
        tr_loader = DataLoader(Subset(ds_train, tr_idx), batch_size=v3.BATCH,
                               shuffle=True, num_workers=v3.NUM_WORKERS)
        va_loader = DataLoader(Subset(ds_eval, va_idx), batch_size=v3.BATCH,
                               shuffle=False, num_workers=v3.NUM_WORKERS)
        model = v3.build_model(n_classes, args.arch)
        ckpt_path = os.path.join(v3.OUT_DIR, f"fold{fold}_best.pt")

        if fold < args.start_fold:
            # 复用已保存的 best 权重：回评 F1 + 预测测试集
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model = model.to(device)
            f1, _ = v3.evaluate(model, va_loader, device)
            print(f"  [fold{fold}] 复用已存权重，val macro-F1 = {f1:.4f}", flush=True)
            fold_f1s.append(f1)
            fold_history[f"fold{fold}"] = [{"reuse": True, "val_f1": f1}]
        else:
            f1, hist = v3.train_one_fold(model, tr_loader, va_loader,
                                         args.epochs, device, fold)
            fold_f1s.append(f1)
            fold_history[f"fold{fold}"] = hist

        p = v3.predict_proba(model, test_paths, device, eval_tf, tta_tfs, tta=args.tta)
        test_proba_sum = p if test_proba_sum is None else test_proba_sum + p
        del model
        torch.cuda.empty_cache()

    mean_f1, std_f1 = float(np.mean(fold_f1s)), float(np.std(fold_f1s))
    print(f"\n==> {args.kfold} 折交叉验证 macro-F1: {mean_f1:.4f} ± {std_f1:.4f}",
          flush=True)

    final_proba = test_proba_sum / args.kfold
    pred_idx = final_proba.argmax(1)
    pred_labels = [idx_to_class[i] for i in pred_idx]

    tag = f"{args.arch}_{args.img_size}"
    out_csv = os.path.join(v3.OUT_DIR, f"submission-v3-{tag}.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("ID,Category\n")
        for pos in order_idx:
            f.write(f"{test_ids_sorted[pos]},{pred_labels[pos]}\n")
    print(f"submission 已写入: {out_csv}  ({len(canon_order)} 行)", flush=True)

    from collections import Counter
    dist = dict(Counter(pred_labels))
    print("预测类别分布:", dist, flush=True)

    log = {
        "arch": args.arch, "img_size": args.img_size, "epochs": args.epochs,
        "kfold": args.kfold, "tta": args.tta,
        "start_fold": args.start_fold,
        "fold_f1": fold_f1s, "mean_f1": mean_f1, "std_f1": std_f1,
        "pred_dist": dist, "history": fold_history,
    }
    log_path = os.path.join(v3.OUT_DIR, f"run-log-{tag}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"运行记录已写入: {log_path}", flush=True)


if __name__ == "__main__":
    main()
