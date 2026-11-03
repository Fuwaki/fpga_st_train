# FPGA 风格迁移交付

两套包都在这棵树里。梵高是 `tiny_dw`，白盒卡通是 `tiny_dw_guided`。都是实验 checkpoint 的 PTQ，没有再训练。`scripts/package.sh` 会先删掉整个 `deliverables/`，只装梵高两包。白盒包是按文末的拷贝说明放进来的。

## 梵高（`tiny_dw`，v6）

两种部署路径：

| 目录名 | 来源 trial | 用途 |
|--------|------------|------|
| `vangogh` | `film_both`（`encoder=sep`） | 常规 int8 卷积 + InstanceNorm + FiLM |
| `vangogh_lut_dw` | `lut_dw_b4` | 编码器两处 depthwise 为 4bit LUT 真值表，其余同左 |

## 树结构

```
deliverables/
├── README.md
├── fp32/<name>/model.json + model.mpk
├── w8a8/<name>/manifest.json + weights.bin + calibration.json + metrics.json + model.json
├── w4a8/<name>/          （同上，4bit 权重打包）
└── gallery/              对比图（int 列为整数参考解释器）
```

**FPGA 最小集合**：`manifest.json`；LUT 路径下 `weights.bin` 仍含 1×1/深度可分离等卷积码，**LUT 本体在 `manifest.lut_layers`**。

## 推理

```bash
# 整数部署（自包含目录）
fpga_st_train infer --model deliverables/w8a8/vangogh --content <图或目录> --out out/ --mode int

fpga_st_train infer --model deliverables/w8a8/vangogh_lut_dw --content <图> --out out/ --mode int
```

`--mode auto` 会读目录里的 `manifest.json` 走整数图。

## 指标

各目录下 `metrics.json` 为 `scripts/evaluate.py` 产出（fp32 / fake-quant sim / int 三方）。
汇总表：`python3 scripts/summary.py deliverables`（把 `w8a8`/`w4a8` 当根扫描需改路径；或直接读各 `metrics.json`）。

## 重建

```bash
scripts/ptq.sh experiments/tiny_dw_vangogh_film/trials/film_both w8a8
scripts/ptq.sh experiments/tiny_dw_vangogh_film/trials/film_both w4a8
scripts/ptq.sh experiments/tiny_dw_vangogh_lut_dw/trials/lut_dw_b4 w8a8
scripts/ptq.sh experiments/tiny_dw_vangogh_lut_dw/trials/lut_dw_b4 w4a8
scripts/package.sh
```

## 白盒卡通（`tiny_dw_guided`）

同一张图、同一套算力（1145 Mmac，45183 参数）。`whitebox_in` 是这组实验里画质最好的 InstanceNorm；`whitebox_ln` 是同结构的 LayerNorm，给行缓冲用（H/4 上不再要整帧统计）。

| 目录名 | 来源 trial | 主干 |
|--------|------------|------|
| `whitebox_in` | `experiments/tiny_dw_whitebox_ln/trials/j_ref_in` | InstanceNorm |
| `whitebox_ln` | `experiments/tiny_dw_whitebox_ln/trials/a_base` | LayerNorm |

目录布局与梵高相同：`fp32/<name>/` 是 `model.json` + `model.mpk`，`w8a8` / `w4a8` 各有 `manifest.json`、`weights.bin`、`calibration.json`、`metrics.json`、`model.json`。

图上比梵多多两个算子，其余沿用已有 manifest：

- `avg_pool2x`：把已量化的 RGB 做 2×2 平均，再量化到 stem 的输入。全分辨率 guide 仍读原图。
- `layer_norm`：每个像素在通道上做均值方差，再乘 γ、加 β，写到下一个点。`instance_norm` 仍是空间上的归约；这两条白盒图里输入点和输出点是分开的（`source_point` → `point`）。旧梵高 manifest 的 `source_point` 为空，解释器仍按原地写。

整数图不包含反射补边。高和宽都要是 8 的倍数（评测 384，部署 640×480 都满足）；奇数边的 `avg_pool2x` 会直接报错。能导出的拓扑是 H/4 主干、至少一个残差块、H/2 FiLM、带 1×1 对齐的 H/2 guide、全分辨率 3×3 ReLU 相加 guide。pixel-shuffle 和全分辨率堆叠不在这条通道里。

6 张 demo 图、长边 384，`scripts/evaluate.py` 的均值（dB）：

| 包 | 位宽 | sim/fp32 | int/fp32 | int/sim | 权重 |
|----|------|----------|----------|---------|------|
| `whitebox_in` | W8A8 | 38.84 | 38.42 | 40.06 | 42120 B |
| `whitebox_in` | W4A8 | 21.67 | 21.75 | 37.21 | 21060 B |
| `whitebox_ln` | W8A8 | 37.79 | 37.69 | 42.13 | 42120 B |
| `whitebox_ln` | W4A8 | 24.25 | 24.48 | 38.27 | 21060 B |

```bash
fpga_st_train infer --model deliverables/w8a8/whitebox_in --content <图或目录> --out out/ --mode int
fpga_st_train infer --model deliverables/w8a8/whitebox_ln --content <图或目录> --out out/ --mode int
```

重建（不要接着跑 `scripts/package.sh`，它会清掉这两包）：

```bash
scripts/ptq.sh experiments/tiny_dw_whitebox_ln/trials/j_ref_in w8a8
scripts/ptq.sh experiments/tiny_dw_whitebox_ln/trials/j_ref_in w4a8
scripts/ptq.sh experiments/tiny_dw_whitebox_ln/trials/a_base w8a8
scripts/ptq.sh experiments/tiny_dw_whitebox_ln/trials/a_base w4a8
# 再把 model.mpk / quant/<bits>/ 拷到 deliverables/{fp32,w8a8,w4a8}/{whitebox_in,whitebox_ln}/
```
