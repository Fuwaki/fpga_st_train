# v3 — int8 量化与 QAT

日期 2026-09-13 · 承接 `v2-feature-and-lr.md`。
v2 定稿（`runs/v2-lr6/final.mpk`，376,595 参数，4000 步），本版把它变成 **能上 FPGA 的定点模型**。

> **后续修订（v5，2026-09-13）——本版的绝对数字已过时，结论仍然成立。**
>
> 1. 这一版的 PSNR 是在"模拟前向没有建模最后一级 uint8 输出量化"的代码上测的，
>    而导出侧的 head 输出被对称 clamp 到 `[-127,127]`，等于 0.498 以上的高光被削。
>    两边都不对，所以 §3.1 的 48.80 dB 偏乐观。v5 把 head 改成无符号输出并补上模拟，
>    交付数字以 `deliverables/*/metrics.json` 为准。
> 2. 子命令签名变了：`calibrate` / `export-int8` 现在读的是**模型目录**（`--model <目录>`）
>    而不是训练 config；旧的 `student-stylize` 换成 `infer`。
> 3. `runs/v2-lr6/` 与 `runs/quant/` 已按用户要求清理，文中路径仅作历史记录。

---

## 1. 量化方案（为什么是这个方案）

目标是"给定点加速器用"，所以方案是按硬件数据通路选的，而不是按框架方便选的：

| 项目 | 选择 | 理由 |
|---|---|---|
| 激活 | 对称 int8，**每张量一个 scale** | 没有 zero point，数据通路里不需要额外的偏移加法器 |
| 权重 | 对称 int8，**每输出通道一个 scale** | 3x3 卷积里一整行滤波器共用一个乘法器，精度损失却远小于 per-tensor |
| 累加器 | int32 | 最坏情况 3x3x64 = 576 项 127x127 ≈ 9.3e6，远小于 2^31 |
| bias | 导出为 int32：`round(b / (s_x * s_w))` | 累加器全程保持整数，直到 requant |
| 残差相加 | 两个操作数先 requant 到**同一张网格** | 定点加法要求 scale 一致，这是硬约束不是技巧 |
| 输出图 | uint8，scale 1/255，zero point 0 | 最后一级 clamp 到 [0,1] 正好用满 8 bit |

**关键设计：训练用的前向和后端模拟用的是同一份代码。**
`TinyStudent::forward_core(content, Some(scales), observe)` 既做 QAT 训练、又做 int8 模拟、
又是导出的依据。没有"另写一个量化模型"的余地，也就不会出现两边不一致。

- 舍入用 **STE（直通估计器）**：`ste_round(x) = x + (round(x).detach() - x.detach())`，
  前向是 round，反向梯度恒等，所以 fp32 训练和定点模拟可以共用一条路径。
- 权重 scale 每步由当前权重的 per-channel absmax 推出（`fake_quant_weight`）；
  激活 scale 由标定得到后**冻结**。

## 2. 三个新子命令

```bash
# 1) 标定：跑 20 个内容 batch，记录 16 个量化点的 absmax
fpga_st_train --device d0 calibrate --config configs/v2-lr6.json \
    --checkpoint runs/v2-lr6/final.mpk --batches 20 --out runs/quant/v2-lr6-scales.json

# 2) 训练时加 quant.enabled=true + quant.calibration=<上一步的文件>，即 QAT
fpga_st_train --device d0 train --config configs/qat-v2.json

# 3) 导出：int8 权重 + 加速器清单
fpga_st_train --device d0 export-int8 --config configs/v2-lr6.json \
    --checkpoint runs/v2-lr6/final.mpk --calibration runs/quant/v2-lr6-scales.json \
    --out runs/v2-lr6-int8
```

另外 `student-stylize --quant <scales.json>` 会走定点模拟路径，
可以直接把"部署后是什么样"和 fp32 逐像素对比。

**16 个量化点**（`image, enc.1.in, enc.2.in, res.{0..3}.c1/c2.in, up3.0.in, up1.0.in,
up3.1.in, up1.1.in, head.in`），残差块的输出网格**复用下一个消费者**的输入索引，
所以一个 block 不需要自己的网格。

## 3. 结果

### 3.1 训练后量化（PTQ）几乎无损

`v2-lr6/final.mpk` 用 20 个 batch 标定后，直接跑定点模拟，与 fp32 逐像素比：

| 图像 | PSNR(int8 vs fp32) | 最大单像素误差 |
|---|---|---|
| animal | 48.52 dB | 9/255 |
| cat | 48.82 dB | 7/255 |
| city | 46.08 dB | 11/255 |
| landscape | 49.87 dB | 6/255 |
| portrait | 49.41 dB | 7/255 |
| water | 50.08 dB | 6/255 |
| **平均** | **48.80 dB** | ≤11/255 |

任务指标也基本没动（animal 纹理 0.00060 → 0.00062，`checker` 仍在 1e-5 以下）。
**结论：对这个模型，int8 量化本身几乎没有代价。**

### 3.2 QAT

`configs/qat-v2.json`：从 v2-lr6 热启动，冻结标定得到的激活 scale，
lr 5e-5 线性衰减到 5e-6，2000 步（14 分钟）。

| 对比 | 平均 PSNR |
|---|---|
| QAT 模型 int8 vs 它自己的 fp32 | **49.37 dB** |
| QAT 模型 int8 vs 原 fp32 参考 | 38.57 dB |
| QAT 模型 fp32 vs 原 fp32 参考 | 38.72 dB |
| PTQ int8 vs 原 fp32 参考 | 48.80 dB |

读法：QAT 之后的量化保真度（49.37 dB）和 PTQ（48.80 dB）**基本持平**——
因为 PTQ 本来就没有损失可捡。38.x dB 那一行说明 QAT 的 2000 步把模型**移动**了
（训练损失从 0.0534 降到 0.0503，等于又白捡了一点训练量），但这不是量化带来的。

**诚实结论：int8 这件事上，PTQ 就够了，QAT 在这个模型上没有可观的收益。**
QAT 的价值在于（a）训练目标和部署目标完全一致，导出的东西和你训的东西是同一个函数；
（b）往下压到更狠的位宽时才有用——见 3.3。

### 3.3 什么时候 QAT 才有用：故意把范围砍一半

标定时加 `--safety 0.5`，让激活 range 只有实际 absmax 的一半，大量激活被截断：

| 图像 | PTQ int8 | QAT int8（800 步） |
|---|---|---|
| animal | 33.01 dB | 36.79 dB |
| cat | 28.51 dB | 34.22 dB |
| city | 31.35 dB | 32.42 dB |
| landscape | 31.70 dB | 33.80 dB |
| portrait | 27.48 dB | 33.22 dB |
| water | 27.06 dB | 29.57 dB |
| **平均** | **29.85 dB** | **33.33 dB** |

（`configs/qat-agg.json`，同样是冻结 scale，只是 scale 只有该有的一半大）

**+3.5 dB，800 步就拿到了。** 这就是 QAT 的正确使用场景：
量化确实有损的时候，它让网络去适应那个网格。
反过来也说明 3.2 的结论不是"QAT 没用"，而是"**这个模型在 int8 下压根没有余量可捡**"。

顺带一个有操作价值的推论：**先用 `--safety` 做一次收紧标定，看 PTQ 掉多少，
就能判断还需不需要 QAT、以及位宽还能不能再压。** 我们这次测出 int8 的余量很大，
所以真要压成本，应该直接去试 int4 而不是在 int8 上做 QAT。

### 3.4 客观风格评价（VGG Gram）—— 一个必须记下来的负面结果

用 `teacher-eval/` 里的 `scripts/metrics.py`（VGG16，relu1_2/2_2/3_3/4_3 的 Gram 距离，
`gram_walk_pct` = 相对"内容图自身到风格图的距离"改善了多少），6 张 demo 图：

| | gram_mean | walk | lapvar | psnr_vs_content | sat |
|---|---|---|---|---|---|
| MicroAST 教师 | 2.50e-5 | **+9.1%** | 0.00191 | 11.57 | 0.101 |
| 我们的学生 | 3.16e-5 | **−15.6%** | 0.00051 | 10.83 | 0.087 |

（注意协议和 `REPORT.md` 略有差别——这里是在原图上推理再缩到 256，
所以教师的 walk 只有 +9.1% 而不是报告里的 22.2%。**行内对比仍然成立**。）

**读法：形态上（明度/饱和度的整体映射）学生是对的，但在 VGG 特征统计上它并没有把图
往风格方向推。** 和 3.1 的纹理比 27% 是同一件事的两种测量：学生输出比教师平滑得多。

这和学生是"从教师蒸馏"出来这件事不矛盾：输出空间 L1 的最优解是条件均值，
教师输出本身纹理也不多（见 v1 笔记），于是学生学到的就是"去掉纹理的教师"。
要修这个只能换损失（统计量/感知损失），不是换量化方案能解决的。
量化这边可以确定地说：**这个问题不是 int8 引入的**（int8 与原 fp32 相差 49 dB）。

拼图对比：`runs/note/v3-int8-comparison.png`（6 行 × 4 列：内容 / 教师 / 学生 fp32 / 学生 int8）。

## 4. 给 FPGA 的交付物

`runs/v2-lr6-int8/`：

- `weights.bin` — 375,824 个 int8 权重（367 KiB），按层执行顺序拼接
- `manifest.json` — 每层的 `in_channels/out_channels/kernel/stride/reflect_pad`、
  `in_point/out_point`（激活 scale 下标）、`weight_scale[]`、`bias_int32[]`、
  `requant[]`、`weight_offset/weight_count`，外加一份 `graph` 给出完整执行顺序

一层算子的数据通路就是：

```
acc = Σ_{ic,kh,kw} x_int8[ic,y+kh,x+kw] * w_int8[oc,ic,kh,kw]   (int32)
acc += bias_int32[oc]
out_int8[oc] = clamp(round(acc * requant[oc]), -127, 127)
```

其中 `requant[oc] = in_scale * weight_scale[oc] / out_scale`。
执行顺序（`graph`）：3 个 stride 卷积 → 4 个残差块（conv-relu-conv-add）→
**2x 最近邻上采样（纯连线）** → conv-relu → conv-relu → 2x 上采样 → conv-relu → conv-relu →
RGB head → uint8 输出。

## 5. 踩到的两个坑（记下来免得再踩）

1. **定点前向漏了第一次上采样**（1/4 → 1/2），导致定点路径输出尺寸是 fp32 的一半。
   发现方式：把定点模拟和 fp32 输出叠在一起比尺寸。教训是"两条路径必须能逐像素对比"，
   这也是我把定点模拟做成可跑命令而不是只做数值检查的原因。
2. **QAT 一开始是从随机初始化开始训的**：框架当时没有"热启动"这个选项，
   配置里写了 checkpoint 路径也没有任何效果，结果日志看着正常（loss 在降），
   但采样图是纯白——因为随机初始化 + 量化 + 冻结 scale 的组合直接退化了。
   现在补了 `run.init_from`（`configs/qat-v2.json` 用它从 v2-lr6 热启动）。
   教训：采样图一定要看，光看 loss 会漏掉整类错误。
3. **d1（Tesla P4）上单次推理会触发 burn-fusion 的 panic**
   （`Ordering is bigger than operations`）。训练循环在 d1 上跑几千步没问题，
   但一次性的 `student-stylize` 偶发崩溃。d0 没出现过。评估统一用 d0。

## 6. 下一步

1. **往下压位宽**：int4 权重 / int8 激活，或者权重也改成 per-tensor。
   在 3.1 的结论下，这一步之前应该先做 3.3 那种"故意收紧"的标定来确定余量。
2. **量化点再合并**：目前 16 个激活 scale 里有几个（`res.*.c2.in`）动态范围很小，
   可以考虑和相邻点共用 scale，省掉一路 requant 乘法器。
3. **导出 int4/混合精度的 manifest**，格式已经按"每层一组 scale"设计，扩展是加字段而不是改结构。
