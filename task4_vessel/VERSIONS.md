# 代码版本留档

这里保存每轮迭代的**完整可运行代码快照**（各自带 `src/`、`tests/`、`README.md`、`requirements.txt`、批处理脚本）。

| 目录 | 对应轮次 | 主要内容 |
|---|---|---|
| `v3_pseudo_full/` | 第一轮改进 | v2 模型（SE + 深监督 + clDice + 差分学习率 + 光照/暗角/JPEG 增强）；clDice 预热、`--full` 全量模型、`make_submission` 自动阈值、`pseudo_label.py`、冒烟测试 |
| `v4_edge_seeds_check/` | 第二轮改进 | 边界加权 BCE（`--edge_w`）、多随机种子（`--seeds`）、`check_submission.py`、`--val_tta`、`--min_size/--close` |
| `v5_thick_gt_tile/` | 粗 GT 修正 + 分块推理 | 评测 GT 约定修正（`--gt_thr 0`，PIL `>0`）、`--init_from` 微调、`--seed_start`、串行低显存推理、读图重试、奇数尺寸 padding、**分块推理 `--tile`**；实测榜分 0.84403 |
| `v6_1024_blend/` | 分辨率链 + 四路融合（最终） | 可配置分辨率 `VESSEL_SIZE`（768/1024 推理后自动缩回 512）、`blend_probs.py` 概率融合、`--extra_dir` 外部数据接口、`eval_tile.py`；实测榜分 0.84803，`submission_final5.csv` 预期 0.850~0.852 |

`v6_1024_blend/` 与当前 `retina/src` 逐文件一致（快照时点），所有版本均通过语法检查，`tests/smoke_test.py` 可在 CPU 上 20 秒内通过。

## 怎么用

- **正常训练/提交**：用 `retina/` 根目录（`runs/`、`pred/` 都在那里，权重和提交文件不随版本目录复制）。
- **想用某一版代码**：直接进入对应版本目录运行即可（它有自己的 `src/`），但默认会在该版本目录下新建 `runs/`、`pred/`。
  如果想复用根目录已有的权重，先设置环境变量：

  ```powershell
  $env:VESSEL_WORK = "C:\Users\zhaopangpang\Documents\deepseek-harness\default-workspace\retina"
  cd versions\v3_pseudo_full
  python src\make_submission.py
  ```

- **对比两版差异**：`v3 -> v4` 只动了 `src/common.py`（新增 `edge_weighted_bce`）、`src/train.py`（`--edge_w/--val_tta`）、`src/run_all.py`（`--seeds`）、`src/infer.py`（生成后自动校验）、`src/make_submission.py`（新增两个透传参数）、`tests/smoke_test.py`，并新增 `src/check_submission.py`。

## 说明

- 更早的 v1（最初基线）和 v2（SE+深监督+clDice 首次引入）没有单独快照；它们的训练配置可以用参数复现：
  - v1 等价配置：`--se 0 --aux 0 --cldice 0 --enc_lr_scale 1.0`（增强里会多出光照/暗角/JPEG，其余一致）。
  - v2 配置：v3 基础上 `--cldice_warmup 0`、不加 `--full`。
- 现有 `runs/r34_f0_b`、`runs/r34_f1`、`runs/r34_f2` 是 v1 结构训练的，v3/v4 的推理代码都能直接加载（按 `model_cfg` 缺省 `se=False, aux=False`）。
