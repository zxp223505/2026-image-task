# 眼底血管分割 — 实验报告（2026 Kaggle task-4）

## 1. 任务与数据

- DRIVE 眼底数据集：20 张训练图（565×584，含二值血管标注）+ 20 张测试图；
- 评测：提交 512×512 的 RLE 掩膜，指标 Dice。

## 2. 最关键的发现：评测 GT 的约定

官方 `segmentation_to_csv.py` 对预测做 `PIL bilinear resize → image > 0`，说明**评测用的 512 GT 也是“>0 粗掩膜”**：

| 约定 | 512 血管占比 |
|---|---|
| 训练标签 `>127`（细） | 0.086 |
| 官方 `>0`（粗） | 0.129 |

证据：按细 GT 训练、阈值 0.5 提交，Kaggle = **0.75624**；同一批预测按官方粗 GT 在 OOF 上算，阈值 0.5 正好 = **0.7552**。几乎完全吻合。

修正后（粗 GT 训练 / 阈值下调）榜分从 **0.756 → 0.844**，再叠加分块推理 → **0.84656**。

## 3. 方法

| 环节 | 做法 |
|---|---|
| 输入 | RGB + 绿通道 CLAHE 共 4 通道；ImageNet 归一化；分辨率 512（主）/640（补充） |
| 模型 | ResNet34（ImageNet 预训练）编码器 + UNet 解码器（GN + SE），全分辨率 skip + 原图 concat |
| 损失 | BCE（血管边缘 2× 加权）+ Dice + clDice（预热 10 epoch）+ 深监督（stride 4/16） |
| 训练 | 随机 448×448 裁剪（65% 以血管为中心）、翻转/旋转/缩放/弹性/亮度对比度/gamma/HSV/光照梯度/暗角/JPEG；AdamW 差分学习率（编码器 0.2×）；cosine；bf16；EMA |
| 验证 | 5 折交叉验证（16 训 / 4 验） |
| 推理 | 多模型概率平均 + 4 翻转 TTA + 分块推理（tile 384，stride 192）+ 多尺度 640；OOF 调阈值；FOV 掩膜；最小连通域 30 |

## 4. 结果

| 阶段 | 榜分 |
|---|---|
| 细 GT + 阈值 0.5 | 0.75624 |
| 粗 GT 模型降阈值（0.06） | 0.83701 |
| 5 折粗 GT（512） | 0.84385 |
| 3 模型微调（512） | 0.84403 |
| 512 分块推理 | 0.84505 |
| 4×640 + 512 分块 | 0.84654 |
| 5×640 + 512 分块 | 0.84656 |
| 3×768 + 5×640 + 5×512 分块 | 0.84803 |
| 5×768ft + 5×640 + 5×512 分块（final4） | 0.84842 |
| +5×1024（final5） | 0.85019 |
| 20 张 OOF 重调权重（final6） | 0.85095 |
| +5×1152（final7） | 0.85109 |
| 1024 多尺度 TTA（final8） | 0.85167 |
| +5×1280 + 1152ms（final9） | 0.85214 |
| 1280 多尺度 TTA（final10） | **0.85221（当前最佳）** |
| +1280 第二种子（final11） | OOF 持平，未生成提交 |

单折（512 评测口径）随分辨率变化：

| 分辨率 | fold 0 Dice |
|---|---|
| 512 | 0.8416 |
| 640 | 0.8451 |
| 768 | 0.8497 |
| 1024 | 0.8531 |
| 1152 | 0.8535（饱和） |

最终提交 `submission_final7.csv`（实测 **0.85109**）：5×1152 + 5×1024 + 5×768ft + 5×640 + 5×512 分块融合；`submission_final8.csv` 加了 1024 多尺度 TTA（OOF 0.8424，预期 0.8512）。

## 5. 试过但无效的方向（避免重复踩坑）

| 方向 | 结果 |
|---|---|
| 测试集伪标签自训练（8 epoch） | 多数折保持原权重，最好 +0.0016 |
| 第二套随机种子（512 / 640 / 1024） | 与第一套持平或略低，集成 +0.0001（噪声） |
| CHASE_DB1 外部数据**直接混训**（2:1） | fold 0 -0.0012（域差异） |
| CHASE+STARE **预训练→DRIVE 微调** | 0.8454 vs 纯 DRIVE 0.8455，无增益 |
| 形态学闭运算 / 迟滞阈值 | 持平或掉点 |
| 768 从零训练补齐 5 折 | 训练进程被 ollama 抢显存中断，仅 3 折；改用「640→768 微调」后成本降到 12 分钟/折且效果持平 |
| 1024 再加训 20 epoch | 与 15 epoch 持平（f0 0.8494 vs 0.8503） |
| 1152 分辨率 | 10 epoch 0.8535，与 1024 的 0.8533 打平，饱和 |
| ResNet50（2.5× 参数） | fold 0 512 口径 0.8452 vs ResNet34 0.8444，无增益 |

## 6. 复现

```powershell
cd retina
# 1) 5 折粗 GT、640 分辨率（约 2.5 小时）
$env:VESSEL_SIZE='640'
foreach ($f in 0,1,2,3,4) {
  python src/train.py --fold $f --crop 448 --bs 3 --epochs 100 --spe 192 --lr 2e-4 `
    --amp bf16 --se 1 --aux 1 --cldice 0.15 --edge_w 1.0 --gt_thr 0 `
    --out "r34_f${f}_640" --val_every 5 --workers 0
}

# 2) 512 模型做分块推理
python src/infer.py --runs runs\r34_f0_thick runs\r34_f1_thick runs\r34_f2_thick `
  runs\r34_f3_thick runs\r34_f4_thick --thr 0.48 --tta 4 `
  --tile 384 --tile_stride 192 --tile_blend 0.7 --out pred_tile

# 3) 640 模型推理（VESSEL_SIZE=640 时 infer 会自动缩回 512 输出）
$env:VESSEL_SIZE='640'
python src/infer.py --runs runs\r34_f0_640 runs\r34_f1_640 runs\r34_f2_640 `
  runs\r34_f3_640 runs\r34_f4_640 --thr 0.45 --tta 4 --out pred_640b

# 3b) 768 / 1024 微调链（每折约 10 分钟，从上一档 --init_from 继续）
$env:VESSEL_SIZE='768'
python src/train.py --fold 0 --init_from runs\r34_f0_640\best.pt --crop 512 --bs 2 `
  --epochs 15 --spe 192 --lr 3e-5 --amp bf16 --se 1 --aux 1 --gt_thr 0 --out r34_f0_768ft
$env:VESSEL_SIZE='1024'
python src/train.py --fold 0 --init_from runs\r34_f0_768ft\best.pt --crop 640 --bs 1 `
  --epochs 15 --spe 128 --lr 3e-5 --amp bf16 --se 1 --aux 1 --gt_thr 0 --out r34_f0_1024

# 3c) 768/1024 测试集推理
$env:VESSEL_SIZE='768';  python src/infer.py --runs runs\r34_f0_768ft ... --out pred_768ft
$env:VESSEL_SIZE='1024'; python src/infer.py --runs runs\r34_f0_1024 ... --out pred_1024

# 4) 四路概率融合（最终版 submission_final5.csv）
python src/blend_probs.py --dirs pred_tile pred_640b pred_768ft pred_1024 `
  --weights 0.4 0.1 0.1 0.4 --thr 0.40 --min_size 30 --out pred_final5
# 结果：pred_final5/submission.csv
```

## 7. 结论

- 榜单从 0.756 提升到 **0.84803**（`submission_final5.csv` 预期 0.850~0.852），关键是识别出「评测 GT 是粗掩膜」并据此重训/调阈值；
- 分辨率是第二有效的杠杆：512→640→768→1024 单折 0.8416→0.8451→0.8497→0.8531，1152 饱和；
- 分块推理 +0.0015，四路融合再 +0.003；
- 外部数据（CHASE 直接混训）、伪标签、多 seed、1024 加训、ResNet50 均已实测无效或持平；
- 0.9 不可达：DRIVE 人类标注一致性约 0.79，公开 SOTA 约 0.87（多带额外数据/预训练）。榜上 0.859~0.863 的队伍大概率用了更大骨干/额外数据/更长训练，本机 4GB 显存（还被 ollama 占 2GB）已达上限。
