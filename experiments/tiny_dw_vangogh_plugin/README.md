# tiny_dw_vangogh_plugin

**问题**：在容量实验胜出的解码器上（`base 32 / res_blocks 4 / mid 48 / fine 24`），单独插入一个轻量模块、并且只插在一个分辨率，1000 step 后验证损失、相对教师 PSNR、lapvar 能不能超过不加模块的基线？

损失、学习率、数据与 `configs/dw_plugin_base.json` 相同：像素 L1 + 2×梯度，lr 6e-4 常数，1000 step。`vangogh_dw` 用 batch 4 才训 4000 step；现在 batch 16，1000 step 看到的样本数相同（都是 1.6 万张）。容量实验在这个步数上，健康配置最后 200 step 的验证损失只动了 0.001–0.006，最好的 `b32_dec48_24` 从 600 step 起就是最低并保持到结束。

## 模块

形状保持的模块在三个位置各跑一次：`h4`（残差干之后、第一次上采样之前，64 通道）、`h2`（H/2 点卷积之后，48 通道）、`full`（全分辨率点卷积之后、head 之前，24 通道）。`shuffle` 只替换第一次最近邻上采样，不看 site。

能做成恒等初始化的模块，step 0 与基线相同。`in`、`ln`、`simam` 一上来就会改激活，这是模块本身。

| plugin | 做什么 | 初始化 |
|---|---|---|
| `00_base` | 不加模块 | — |
| `affine` | 逐通道 γ、β。固定风格下 FiLM 剩下的常数部分 | γ=1，β=0，恒等 |
| `in` | 空间 InstanceNorm 再乘 γ 加 β。教师 AdaIN 在风格固定后的样子 | 归一化立即生效 |
| `ln` | 每个像素上对通道做 LayerNorm，再 γ、β。不需要整图归约，可流式 | 归一化立即生效 |
| `se` | Squeeze-and-Excitation，压缩比 4 | 末层为 0，`2·sigmoid(0)=1` |
| `film` | 全局池化 → 小瓶颈 → γ、β | 两个头为 0，起步是恒等 |
| `eca` | 全局池化向量上的环形 1 维卷积（ECA-Net） | 卷积核为 0，同样 `2·sigmoid` |
| `grn` | ConvNeXt V2 的全局响应归一化 | γ=0、β=0，起步是恒等 |
| `ca` | Coordinate Attention：沿 H、沿 W 分开池化，门控可以看位置 | 两个输出卷积为 0 |
| `frelu` | `max(x, DW3×3(x))`，给激活一个局部空间条件 | 深度卷积核是恒等 |
| `dw7` | 残差 7×7 深度卷积，覆盖 4–8 像素笔触尺度 | 权重为 0，起步是恒等 |
| `simam` | 无参数的 SimAM 能量门控 | 无参数，立即改变激活 |
| `shuffle` | 分组 1×1（每通道复制 4 份）+ pixel-shuffle，代替第一次最近邻 | 恒等时等于最近邻 |

`h4` 上的归一化对应教师把 AdaIN 放在上采样之前。`h2` / `full` 上的 `in` 是故意留的对照：最近邻之后再归一化，以前会放大 2 像素棋盘格，`checkerboard` 若从 1e-6 升到 1e-2 以上，这个 trial 作废。

整数量化导出还不认识这些算子。带 plugin 的 trial 可以训练、可以用 fp32 推理；`export` 会直接拒绝。选出胜者后再补进 manifest。

## 命令

```bash
python scripts/study_gen.py experiments/tiny_dw_vangogh_plugin
# 同一步只跑一次教师前向，再给 3 个学生反传。35 个 trial 分成 12 组。
GROUP=3 scripts/study_run.sh tiny_dw_vangogh_plugin d0
```

容量实验里同结构 1000 step 大约 1.7 小时。日志在 `experiments/tiny_dw_vangogh_plugin/log/`。一组三个学生的 stdout 写在 `log/<a>+<b>+<c>.log`，每个 trial 自己的 `metrics.jsonl` 仍在 `trials/<id>/`。

先看 H/4 时只跑 `00_base`、`shuffle` 和 `*_h4` 即可，共 13 个，大约一天。短烟测：

```bash
ITERATIONS=20 scripts/study_run.sh tiny_dw_vangogh_plugin d0
```

跑完后：

```bash
python scripts/study_collect.py experiments/tiny_dw_vangogh_plugin
```

和基线比三列：验证损失、三张采样图的平均 `psnr_vs_teacher`、`lapvar / lapvar_teacher`。基线在 1000 step 时是验证损失 0.210、PSNR 18.46 dB、lapvar 约 0.00041（教师约 0.006）。这轮损失仍然不奖笔触，lapvar 不动而验证损失下降，只说明模块更会拟合颜色和粗边缘。
