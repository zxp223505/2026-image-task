# 植物分类（课设任务一）进度记录

- **截止时间**：2026-10-10 24:00，成绩按排行榜排名
- **最后更新**：2026-09-28 晚
- **数据**：`C:\Users\zhaopangpang\Desktop\2026task1\dataset-for-task2\dataset-for-task2`
  （train 5 类 × 100 张 = 500，test 378 张，无标签）
- **环境**：torch 2.14.0+cu130，RTX 500 Ada Laptop GPU（**仅 4GB 显存**）

---

## 一、发现的三个问题（v1/v2 都中招）

### 1. 验证集误用训练增强（最严重）

```python
full_ds = ImageFolder(TRAIN_DIR, transform=train_tf)   # 带随机裁剪/翻转/颜色抖动
...
va_loader = DataLoader(Subset(full_ds, va_idx))        # Subset 不换 transform！
```

`Subset` 不会替换 transform，所以验证集也在做随机增强。后果是 `val_f1`
每次都不一样，而 `best checkpoint` 正好按这个噪声指标挑 —— 等于闭眼选权重。
**推测这就是 v2 某折只有 0.64 的原因**（把糟糕 epoch 的权重当成 best 存了）。

### 2. 提交文件行顺序与官方样例不一致

官方 `submission-for-task2.csv` **不是字典序**，而 v1/v2 用的是
`sorted(os.listdir(TEST_DIR))`。两个文件 ID 集合相同但顺序不同。
若助教脚本按行位置比对 → 直接零分。v3 已改为读样例顺序输出。

### 3. 产物路径依赖 cwd

v1/v2 保存 checkpoint 用的是相对路径 `f"fold{fold}_best.pt"`，
存到哪取决于从哪个目录运行。

---

## 二、干净验证集回评结果（已完成，无重训）

脚本 `reeval_baselines.py`，用 `eval_tf` 确定性变换重新评估已有 checkpoint，
同一折划分（`random_state=42`），分辨率按各自训练时（v1=224 / v2=288）：

| 模型 | macro-F1 | 折数 | 各折明细 |
|---|---|---|---|
| **v1 ResNet18@224** | **0.8638 ± 0.0406** | 5 | 0.8201 / 0.8151 / 0.9193 / 0.8919 / 0.8726 |
| v2 ResNet50@288 | 0.7487 ± 0.1088 | 2 | 0.8575 / **0.6399** |

结论：**ResNet18 反而赢 ResNet50 11.5 个点**。但 v2 的对比不可全信 ——
它的 `fold0_best.pt`(10:01) 和 `fold1_best.pt`(09:55) **来自两次不同的运行**，
且 checkpooint 是被污染指标选出来的。

**v1 的短板（fold4 逐类 F1）**：

```
Black-grass         0.6667   ← recall 仅 0.55，瓶颈所在
Common wheat        0.9302
Loose Silky-bent    0.7907
Scentless Mayweed   0.9756
Sugar beet          1.0000   ← 已完全分开
```

后续提分收益应主要来自 **Black-grass**（主要被 Loose Silky-bent 蚕食）。

详细数据见 `baseline-reeval.json`。

---

## 三、当前状态：**训练已中断，明天重跑**

运行到 `fold0 epoch 5/25`（val_f1=0.8077）时手动停止。

> ⚠️ `fold0_best.pt` 是这次中断运行的产物，**不完整、不可用于提交**。
> 目录下没有生成任何 submission CSV。明天重跑会覆盖它。

实测速度 **约 1.1 分钟/epoch**，全量 5 折 × 25 epoch ≈ **2.2~2.5 小时**。

---

## 四、明天从这里继续

```bash
cd C:\Users\zhaopangpang\Desktop\2026task1-v3

# 第 1 步：ResNet18@224（与 v1 同架构同分辨率，验证修复无回归）
python -u plant_classifier_v3.py --arch resnet18 --img-size 224 > train-resnet18-224.log 2>&1

# 第 2 步：跑完第 1 步再跑这个（4GB 显存装不下两个并行任务）
python -u plant_classifier_v3.py --arch resnet50 --img-size 288 > train-resnet50-288.log 2>&1
```

跑完会生成：
- `fold0~4_best.pt`
- `submission-v3-<arch>_<size>.csv`（行序已对齐官方样例）
- `run-log-<arch>_<size>.json`（每折每 epoch 的 F1，可看收敛曲线）

**对比靶子**：v1 的 `0.8638 ± 0.0406`。v3 ResNet18 打平或更高 = 修复干净。

### 之后的待办

1. 两个架构都跑完后做**概率平均集成**，再看 TTA 增益
2. 提交到排行榜验证（测试集无标签，本地数字只是代理指标）
3. 若还有时间：针对 Black-grass 做 class weight 或专门的数据增强

---

## 五、脚本说明

| 文件 | 用途 |
|---|---|
| `plant_classifier_v3.py` | 主训练脚本。架构/超参/增强策略与 v2 一致，只修了上面三个问题 |
| `reeval_baselines.py` | 回评已有 checkpoint，输出 `baseline-reeval.json` |
| `train-resnet18-224.log` | 中断的那次运行日志 |
| `进度.md` | 本文件 |

**约束**：保持 ResNet18/ResNet50 架构不变，不改动 `Desktop\2026task1` 里的任何原文件。
