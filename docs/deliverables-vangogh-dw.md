# FPGA 风格迁移 — `tiny_dw` 梵高交付（v6）

两种部署路径，均来自实验 checkpoint 的 PTQ，无再训练：

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
