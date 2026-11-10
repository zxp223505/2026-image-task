# 眼底图像血管分割（Kaggle 2026task-4）

DRIVE 眼底数据集：20 张训练图（565×584，含二值血管标注）+ 20 张测试图，按 512×512 分辨率上的 Dice 评分。

## 快速运行（Windows 双击）

在资源管理器里进入 `retina\`，双击：

| 文件 | 作用 | 耗时 |
|---|---|---|
| **`一键生成提交.bat`** | 调 `make_submission.py`：用已训好的权重推理（自动采用 OOF 调好的阈值），生成 `pred\submission.csv` | 约 1~2 分钟 |
| **`一键训练并提交.bat`** | 完整训练 5 折（v2 模型）后再集成推理 | 约 2 小时（每折约 25 分钟） |

用 PyCharm 的话：直接右键运行 **`src/make_submission.py`**（无参数，自动找 `runs/*` 下所有 checkpoint 集成，输出 `pred/submission.csv`；若 `runs/oof_summary.json` 存在会自动采用里面调好的阈值，否则用 0.5）。
`src/run_all.py` 是完整流程入口（默认会训练；加 `--skip_train` 只做评估+推理），`src/infer.py` 单独跑必须带 `--runs` 参数。

运行结束后窗口会停住并显示提交文件路径。中途关掉训练窗口后，可在命令行用 `python src\run_all.py --resume` 断点续训。
（如果双击后提示找不到 `python`，说明 Python 没在 PATH 里，改用 `python` 的绝对路径执行同一条命令即可。）


## 0. 先说一个关键坑：提交的标签约定

比赛说明写“血管像素置 0、其他 255”，但 **`train/label/*.jpg` 的实际约定是血管 = 255（白）、背景 = 0（黑）**（已核对像素直方图，血管占比约 8%~11%，与 DRIVE 标注一致）。
官方 `segmentation_to_csv.py` 的逻辑是 `image[image > 0] = 1` 后对 1 做 RLE，即编码**白色像素**。
所以预测图必须存成 `血管=255 / 背景=0`。若按说明把血管画成 0，RLE 编码的就是背景，Dice 会接近 0。
`src/infer.py` 直接按官方同样的 RLE 规则生成 CSV，已与官方脚本逐字符比对一致。

### ⚠️ 更关键的发现：评测用的 512 GT 是「>0 粗掩膜」

官方转换脚本对预测图做的是 `PIL bilinear resize -> image > 0`，评测 GT 也按同样方式生成：
512 分辨率下血管占比 **0.129**，而按 `>127` 只有 **0.086**（粗 1.5 倍）。

证据：按细 GT 训练、阈值 0.5 提交，Kaggle 得分为 **0.75624**；用官方同款粗 GT（PIL bilinear 后 >0）在 OOF 上算，同一批预测在阈值 0.5 下正好是 **0.7552**。几乎完全对上。

**后果与对策（代码已按此调整）**：
- `train.py` / `run_all.py` 的 `--gt_thr` **默认已改为 0**（PIL 复现官方 resize），新训练直接学粗 GT，阈值用 0.5 左右；
- 手上已有的**细 GT 模型**两种用法：把阈值降到 **0.04~0.06**，或者用 `--init_from` 在粗 GT 上微调 15 epoch（实测单折 +0.03~0.05，集成 OOF 0.8315）。

```powershell
# A. 不重训：用官方粗 GT 重新调阈值（细 GT 模型）
python src/oof_eval.py --runs "runs/r34_f*" --gt_thr 0
python src/make_submission.py --thr 0.06

# B. 在粗 GT 上微调已有权重（约 4 分钟/折）
python src/train.py --fold 0 --init_from runs\r34_f0_b\best_ema.pt `
  --gt_thr 0 --epochs 15 --lr 5e-5 --spe 192 --bs 4 --amp bf16 --workers 0 --out r34_f0_thick

# C. 从头用粗 GT 训练 5 折（默认 --gt_thr 0）
python src/run_all.py
```

## 1. 方法

| 环节 | 做法 |
|---|---|
| 预处理 | 缩放到 512×512（与评测一致）；输入 4 通道 = RGB + 绿通道 CLAHE；ImageNet 归一化 |
| 模型 | ResNet34（ImageNet 预训练）编码器 + UNet 解码器（GN + SE），全分辨率 skip，输出层 concat 原图 |
| 训练 | 512 图随机 384×384 裁剪（65% 以血管像素为中心）；翻转、任意角旋转、缩放、弹性形变、亮度/对比度/gamma/HSV/光照梯度/暗角/JPEG 压缩；BCE + Dice + **clDice** + 深监督；AdamW（编码器差分学习率）+ cosine；bf16；EMA |
| 验证 | 5 折交叉验证（16 训练 / 4 验证），512 全图 + TTA |
| 推理 | 多折 EMA（可加 raw）概率平均，4 方向翻转 TTA（可选 4×90° 旋转、多尺度）；FOV 视野掩膜；OOF 上调阈值；形态学后处理默认关闭（实测掉点，可用 `--close/--min_size/--hyst_lo` 再试） |

`bf16` 是刻意选的：fp16 + GradScaler 在 4GB 卡上训练到第 9 个 epoch 左右会数值溢出并把权重打成 NaN；换 bf16 后完全稳定（代码里也保留了 loss/grad 非有限则跳过该 step 的保护）。

### v2 相比 v1 的改动（都可通过参数关闭）

| 改动 | 作用 | 开关 |
|---|---|---|
| clDice 软骨架损失 | 细血管拓扑/连通性，DRIVE 上通常 +0.005~0.015 | `--cldice 0` 关闭 |
| clDice 预热 10 epoch | 训练初期骨架未成形时不让拓扑项干扰 | `--cldice_warmup 0` 关闭 |
| 边界加权 BCE | 1px 血管边缘权重 ×2，改善细血管轮廓 | `--edge_w 0` 关闭 |
| 深监督（stride 4/16 辅助头） | 让浅层解码器也收到监督，细结构更稳 | `--aux 0` 关闭 |
| 解码器 SE 通道注意力 | 重新校准血管/背景特征 | `--se 0` 关闭 |
| 编码器差分学习率（0.2×） | 预训练特征少被破坏，小数据更稳 | `--enc_lr_scale 1.0` 关闭 |
| 光照梯度 / 暗角 / JPEG 伪影增强 | 模拟手机/相机与压缩差异 | 始终开启 |
| 多随机种子（每折 N 个） | 更多模型做集成，降低单次训练方差 | `--seeds 2` |
| `last.pt` + `--resume` | 断点续训，防 4GB 卡长训练中断 | `--resume` |

**向后兼容**：新模型结构参数（`se/aux`）会写进 checkpoint 的 `model_cfg`。v1 训好的 `runs/r34_f*` 没有该字段，`infer.py/eval_variants.py` 会按老结构（`se=False, aux=False`）加载，不会因缺 key 报错。

## 2. 已验证结果（512 Dice）

| 模型 / 提交 | 数据 | Dice |
|---|---|---|
| **Kaggle：`submission_thickFT.csv`（3 模型微调，thr 0.44）** | 测试 20 张 | **0.84403** |
| Kaggle：`submission_thick5.csv`（5 模型，thr 0.46） | 测试 20 张 | 0.84385 |
| Kaggle：`submission_thickGT.csv`（细 GT 模型，thr 0.06） | 测试 20 张 | 0.83701 |
| Kaggle：细 GT 模型 + 阈值 0.5（修正前） | 测试 20 张 | 0.75624 |
| OOF：5 折粗 GT，全图推理 | 20 张 | 0.8352 |
| OOF：5 折粗 GT，**分块推理 `--tile 384`** | 20 张 | **0.8363** |
| OOF：两套种子 10 模型（第二套偏弱） | 20 张 | 0.8347 |

关键结论：**评测 GT 是 `PIL bilinear >0` 的粗掩膜**（血管占比 0.129 vs 训练标签 `>127` 的 0.086），修正后榜分从 0.756 → 0.844。分块推理（匹配训练时的 384 裁剪尺度）再 +0.0016。第二套随机种子没有增益（fold1/2/4 更弱）。

## 3. 怎么跑

```powershell
# ===== 当前最佳流程（粗 GT + 分块推理）=====
# 已有 5 个粗 GT 模型时，直接出提交（OOF 0.8363）
python src/infer.py --runs runs\r34_f0_thick runs\r34_f1_thick runs\r34_f2_thick runs\r34_f3_thick runs\r34_f4_thick `
  --thr 0.48 --tta 4 --tile 384 --tile_stride 192 --tile_blend 0.7 --out pred_tile

# 从零训练 5 折（默认 --gt_thr 0 = 粗 GT）
python src/run_all.py

# 在已有细 GT 权重上微调成粗 GT（约 4 分钟/折）
python src/train.py --fold 0 --init_from runs\r34_f0_b\best_ema.pt --gt_thr 0 `
  --epochs 15 --lr 5e-5 --spe 192 --bs 4 --amp bf16 --workers 0 --out r34_f0_thick

# 再加一个"全部 20 张训练图"的模型进集成（略涨点，多约 20 分钟）
python src/run_all.py --full

# 只训练第二套随机种子（seed 1），不重跑 seed 0
python src/run_all.py --seeds 2 --seed_start 1

# 已有权重时只做评估 + 推理
python src/run_all.py --skip_train --thr 0.50 --tta 4

# 断点续训（每 10 epoch 存 last.pt）
python src/run_all.py --resume

# 单折训练
python src/train.py --fold 0 --epochs 100 --spe 192 --arch resnet34 --crop 384 --bs 4 `
  --lr 2e-4 --enc_lr_scale 0.2 --se 1 --aux 1 --cldice 0.15 --amp bf16 --out r34_f0

# 想复现 v1 基线：关掉新模块
python src/train.py --fold 0 --se 0 --aux 0 --cldice 0 --enc_lr_scale 1.0 --gt_thr 127 --out r34_v1

# OOF 阈值 / 后处理 / 迟滞阈值搜索（粗 GT）
python src/oof_eval.py --runs "runs/r34_f*_thick" --gt_thr 0

# 推理变体对比（raw/EMA、多尺度、旋转 TTA）
python src/eval_variants.py --runs "runs/r34_f*"

# 分块推理 vs 全图推理对比
python src/eval_tile.py --runs runs\r34_f0_thick runs\r34_f1_thick runs\r34_f2_thick runs\r34_f3_thick runs\r34_f4_thick

# 可选：用测试集伪标签自训练（先跑过 make_submission 生成 pred/*_prob.npy）
python src/pseudo_label.py --runs runs/r34_f*
# 然后再出提交（默认会把 *_pl 模型一起集成；不想用就加 --exclude "*_pl"）
python src/make_submission.py

# 20 秒冒烟测试（CPU、不需要数据，验证模型/损失/TTA/后处理代码没坏）
python tests/smoke_test.py

# 校验提交 CSV（RLE 必须有序、正数、不重叠；infer 结束时会自动跑一遍）
python src/check_submission.py pred/submission.csv
```

数据目录默认 `C:\Users\zhaopangpang\Desktop\4\segmentation`，可用环境变量覆盖：

```powershell
$env:VESSEL_DATA = "D:\data\segmentation"
$env:VESSEL_WORK = "D:\work\retina"
```

架构可选 `resnet18 / resnet34 / resnet50 / resnet101`（`--arch`，编码器通道自动适配）。`resnet18` 更快、适合多折快速集成；`resnet50` 更重，4GB 卡建议把 `--bs` 降到 2。不同架构的权重可一起丢给 `infer.py --runs ...` 做异构集成。

环境：Python 3.13 + torch 2.14 (cu130) + torchvision 0.29 + opencv + numpy/pandas/Pillow，见 `requirements.txt`。
显存 4GB 即可（384 crop、batch 4、bf16 时峰值约 2.2GB；开 clDice 和深监督约多 0.3GB）。

## 4. 可选：测试集伪标签自训练（`src/pseudo_label.py`）

测试图和训练图同域，可以用模型自己在测试图上的高置信预测做一轮自训练：

1. 先跑 `make_submission.py`（或 `infer.py`），得到 `pred/<id>_prob.npy`；
2. `pseudo_label.py` 把 `prob>0.85` / `<0.15` 的像素当伪标签，`0.15~0.85` 的像素用权重 0 忽略；
3. 每折模型在「50% 真标签 crop + 50% 伪标签 crop」上以 `3e-5` 小学习率微调 10 个 epoch（EMA 保存）；
4. 结果存到 `runs/<run>_pl/`，`make_submission.py` 会自动把 `*_pl` 一起集成（不想用加 `--exclude "*_pl"`）。

只用了无标注测试图，没有测试集人工标注；属于可选的 transductive 自训练。**实测在 DRIVE 同域数据上几乎没有增益**（8 epoch 后多数折保持原权重，最好的一折 +0.0016），不推荐作为主要提升手段。

## 5. 文件

```
src/common.py          数据/增强/模型（SE、深监督）/指标/clDice/粗 GT（GT_THR）
src/train.py           单折训练（差分 LR、EMA、bf16、NaN 保护、断点续训、--init_from、OOF 输出）
src/infer.py           集成推理 + RLE + submission.csv（多尺度/旋转 TTA、分块推理 --tile、迟滞阈值）
src/oof_eval.py        OOF Dice、阈值与后处理搜索（--gt_thr 切换细/粗 GT）
src/eval_variants.py   raw/EMA、多尺度、旋转 TTA 对比
src/eval_tile.py       全图 vs 分块推理对比
src/pseudo_label.py    可选：测试集伪标签自训练（实测收益很小）
src/make_submission.py 一键出提交（自动发现权重 + 自动用 OOF 调好的阈值）
src/check_submission.py 提交格式校验（RLE 有序/正数/不重叠）
src/run_all.py         一键全流程（训练/评估/推理）
src/viz_test.py        测试图预测可视化
tests/smoke_test.py    CPU 冒烟测试（模型/损失/TTA/后处理）
runs/r34_f*/           各折权重 best.pt / best_ema.pt / last.pt 与 OOF 概率
versions/              代码版本快照（v3/v4/v5）
```

提交（按实测榜分排序）：`submission_thickFT.csv`（0.84403）> `submission_thick5.csv`（0.84385）> `submission_thickGT.csv`（0.83701）；未实测的 `submission_tile.csv`（分块推理，OOF 0.8363）和 `submission_thick5_full.csv`（全 TTA）可作为下一轮尝试。
