# 2026 课程设计 · 四项图像分析任务

本仓库为课程设计报告的配套代码，覆盖四个 Kaggle 任务：图像分类、面部表情识别、
眼底黄斑中心凹定位、视网膜血管分割。**仓库只包含代码与实验记录，不含数据集与模型权重**。

## 结果一览

| 任务 | 问题类型 | 指标 | 本地/交叉验证 | 排行榜（Public / Private） |
|---|---|---|---|---|
| 任务一 植物幼苗分类 | 小样本 5 类分类 | macro-F1 | 0.8862 ± 0.0204（5 折） | 0.9100 / — |
| 任务二 面部表情识别 | 不均衡 6 类分类 | 准确率 | 未留存（见下） | 0.719 / — |
| 任务三 黄斑中心凹定位 | 关键点回归 | RMSE ↓ | 21.78（无偏 OOF） | 16.83763 / **16.20423** |
| 任务四 视网膜血管分割 | 像素级分割 | Dice ↑ | 0.7552（细 GT 口径 OOF） | 0.85221 / **0.85395** |

> 任务二在训练阶段未落盘逐折指标，因此只报告排行榜真实分数，不虚构本地数值。

## 目录结构

```
task1_plant/     植物幼苗分类：ResNet18@224 + 5 折 + TTA
task2_fer/       面部表情识别：ResNet18 基线 + ResNet50 五折 logits 集成
task3_fovea/     中心凹定位：视野圆归一化 + 视盘先验 + 两级热图回归
task4_vessel/    血管分割：ResNet34+UNet、BCE+Dice+clDice、分辨率链、多路融合
```

## 环境

- Python 3.13，PyTorch 2.14（CUDA 13.0），torchvision 0.29
- 硬件：单卡 RTX 500 Ada Laptop，**仅 4 GB 显存**（部分任务需 batch=1~4）
- 任务四额外依赖：`opencv-python`、`pandas`

## 各任务复现要点

### 任务一 植物幼苗分类

```bash
python task1_plant/plant_classifier_v3.py --arch resnet18 --img-size 224
```

5 折 × 25 epoch，TTA 推理，按官方 sample 的行序写出提交文件。
`run-log-resnet18_224.json` 记录了逐折与逐 epoch 的 F1；`baseline-reeval.json` 是对旧版本
checkpoint 的干净验证集回评（v1 ResNet18 0.8638 / v2 ResNet50 0.7487）。
`NOTES.md` 记录了初版的两个实现缺陷（验证集误用训练增广、提交行序与样例不一致）及修复过程。

### 任务二 面部表情识别

```bash
python task2_fer/train_baseline.py     # ResNet18 基线，7 epoch，类别加权交叉熵
python task2_fer/fast_predict.py       # ResNet50 五折 logits 平均 + 提交
```

注意 `fast_predict.py` 中的 `idx2label = {0:0,1:1,2:2,3:5,4:3,5:4}` —— 这是把 ImageFolder 的
字典序类别索引映射到官方编号，修正它带来了 0.709 → 0.719 的提升。

### 任务三 黄斑中心凹定位

```bash
python task3_fovea/prep.py             # 视野圆 + 视盘检测 -> geom.json
python task3_fovea/build_canon.py      # 生成 2000x2000 标准化图像缓存
python task3_fovea/hm1.py --epochs 100 --tag hm1b --aug 1.2   # 第一级：全图热图
python task3_fovea/hm2.py --oof hm1b --tag hm2 --epochs 70    # 第二级：补丁精修
python task3_fovea/collect_final.py --tag hm2 --base hm1b --stage 2
python task3_fovea/final4.py --var hm2=ff --out submission.csv
```

关键设计：视野圆归一化（统一两种画幅）、视盘相对坐标先验（x 标准差从 206 px 降到 59 px）、
峰值门控 soft-argmax、固定 epoch 快照平均（避免 epoch 选择偏差）、与两级不一致度收缩。
`SOLUTION.md` 记录了完整的正/负结果与方法论反思；`late_avg.py` 用于量化 epoch 选择偏差。

### 任务四 视网膜血管分割

```bash
# 5 折、粗 GT（gt_thr=0）、640 分辨率（详见 task4_vessel/REPORT.md）
python task4_vessel/train.py --fold 0 --crop 448 --bs 3 --epochs 100 --spe 192 \
    --lr 2e-4 --amp bf16 --se 1 --aux 1 --cldice 0.15 --edge_w 1.0 --gt_thr 0 \
    --out r34_f0_640
# 分块推理 + 四翻转 TTA
python task4_vessel/infer.py --runs runs/r34_f0_640 ... --thr 0.48 --tta 4 \
    --tile 384 --tile_stride 192 --tile_blend 0.7 --out pred_tile
# 多路概率融合
python task4_vessel/blend_probs.py --dirs pred_tile pred_640b pred_768ft pred_1024 \
    --weights 0.4 0.1 0.1 0.4 --thr 0.40 --min_size 30 --out pred_final5
```

任务四最关键的一步不是模型，而是发现**评测 GT 是粗掩膜**（官方脚本 `resize → >0`），
据此改用粗 GT 训练并下调提交阈值，使榜分从 0.75624 跃升到 0.83701（占总提升的 84%）。
详见 `task4_vessel/REPORT.md`。

## 不包含的内容

- 数据集（DRIVE / 植物幼苗 / FER），体积过大且受赛事条款约束
- 模型权重（`*.pt`）与中间概率图（`*.npy`），合计数十 GB
- 提交用的 CSV 结果文件

## 说明

`task3_fovea/` 与 `task4_vessel/` 中保留了若干**负结果**脚本（如 `anat_constraint.py`、
`cmp_shrink.py`、`experiment.py`），对应报告中"实测无效的方向"小节，便于复查。
