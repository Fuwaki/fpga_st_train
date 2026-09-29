# v5 — 两个风格、三个交付路径

日期 2026-09-13 · 承接 `v4-low-bit.md`。

这一版不是"再调一次模型"，而是把 v1–v4 的结论**收敛成一个能交付的东西**。
用户的指示很明确：

> 只留三个路径（一个 fp32、一个 W8A8、一个 W4A8），其他全部删掉，以前的产物可以删掉，
> 仅仅保留 note。清理之后训练梵高 / ink 两种风格，交付文档里要有参数文件路径，
> 并且完善推理工具。

---

## 1. 目录约定（先定规矩）

前面几版最大的问题是"东西能跑，但找不到"：configs 有 22 个、runs 有 26 个目录、
同一份标定散在四处。这一版先把约定定死，再往里填内容。

```
configs/<风格>.json        训练配置：一个风格一个文件，只有这一处需要 config
runs/<风格>/               训练产物，自带 model.json，所以它本身就是一个模型目录
  model.json               架构 + 选项 + 元信息（风格、步数、run 名）
  model.mpk                最终权重
  step_XXXXXX.mpk          周期性检查点
  sample_*/target_*.png    训练过程中的采样（人看进度用）
  quant/<bits>/            这个风格的导出（见下）
runs/note/                 设计笔记（本目录，唯一保留的历史）
runs/log/<风格>.log        完整训练 stdout
deliverables/              交付，只有三条路径
  fp32/<风格>/             model.json + model.mpk
  w8a8/<风格>/             model.json + manifest.json + weights.bin + calibration.json + metrics.json
  w4a8/<风格>/             同上
  gallery/                 可视化对比
```

**关键点：`runs/<风格>/` 和 `deliverables/*/<风格>/` 结构是同构的。**
两者都带 `model.json`，所以 `fpga_st_train infer --model <任何目录>` 都能直接跑，
不需要知道它是训练产物还是交付产物、也不需要回头找训练配置。

## 2. 两个风格

配方完全没有动，就是 v2 定下来的那套（`1.0*L1 + 2.0*gradL1`，lr 6e-4 恒定，
crop 192，batch 4，4000 步），**只换风格图**。这样做是有意的：v2-lr6 已经被证明是
"效果已经不错"的配方，这一版要验证的是"换个风格它还能不能行"，不是再调一次超参。

| | ink | vangogh |
|---|---|---|
| 风格图 | `styles/ink/fankuan.jpg` | `styles/vangogh/starry_night.jpg` |
| 训练设备 | d0（RX 580 2048SP） | d1（Tesla P4） |
| 步数 / 耗时 | 4000 / 1602 s（2.50 step/s） | 4000 / 1511 s（2.65 step/s） |
| 最终 loss | 0.05278（l1 0.02626，grad 0.01326） | 0.13856（l1 0.06107，grad 0.03874） |
| `psnr_vs_teacher` | ~30.2 dB | ~21.9 dB |
| 教师 lapvar / 学生 lapvar | 0.00102 / 0.00058（57%） | 0.00794 / 0.00407（51%） |

两件事值得记下来：

1. **梵高的 loss 明显更高，这是对的**。starry night 把图改得比水墨狠得多，
   同时它的目标本身对比度高（输出标准差 36.7 LSB vs 17.3 LSB）。用 loss 直接比两个风格
   没有意义。
2. **两个风格都只保留了教师约 50–57% 的拉普拉斯能量**。也就是说学生确实比教师平滑，
   但**没有"低一个量级"**——早先笔记里那句"低一个量级"是在更早、更差的检查点上测的，
   这一版已经修正。保留一半纹理是输出级 L1 蒸馏的正常表现：L1 的最优解是条件均值，
   天然抹高频。

两个风格并行训练（d0 / d1），互不干扰，26 分钟内全部跑完。

## 3. 这一版修掉的三个真问题

### 3.1 head 的输出范围：一个真 bug，v3 的数字偏乐观了

以前模拟前向里**完全没建模最后一级 uint8 输出量化**，而导出的 manifest 里 head 的输出
会被 clamp 到对称 int8 的 `[-127, 127]`，也就是 **0.498 以上的高光全部被削掉**。
两件事加起来：模拟报告的数字比硬件能达到的好，而且硬件那边白丢一半亮度范围。

修法分两步：

1. **head 的输出改成无符号**（`out_unsigned: true`），clamp 到 `[0, 255]`，scale 1/255。
   图像本来就非负，这是唯一一个显然该用满无符号码本的地方，不涉及任何取舍。
2. **模拟前向补上这一级**（`quant::fake_quant_output`：先 clamp 到 [0,1]，再 round 到
   1/255 网格），让"模拟"和"manifest"描述的是同一个图。

代价几乎为零：多一次 1/255 的舍入，W8A8 上大约损失 0.1 dB 量级；
但换来的是**模拟和导出第一次严格对应**（见 §3.3 的 `int vs sim` 指标）。

注意这和 v4 §4 那个"把**所有**激活改成无符号"的失败尝试是两回事：
那次是 16 个量化点全部改无符号，出现了增益/直流偏移错误；
这次只改 head 一个点，而 head 的输出范围本来就是 `[0,1]`，不存在"白扔一半码本"之外的
任何风险。**v4 §4 的问题仍未解决，代码保持对称路径。**

### 3.2 4bit 权重打包：184 KiB，不是 367 KiB

以前 4bit 权重也是一个 code 一个字节，等于没省。现在 `weights.bin` 按位宽打包：

| 位宽 | `weight_packing` | 布局 |
|---|---|---|
| 8 | `i8-x1` | 一码一字节，二进制补码 |
| 4 | `i4-x2` | 一字节两码，**低 nibble 是前一个码**，各自补码（`0xF` = −1） |

187,912 B = 375,824 / 2，正好一半。
**每层的字节位置在 manifest 里显式给出**（`weight_byte_offset` / `weight_bytes`），
不用 RTL 那边自己算偏移——这类"你自己推一下"的地方正是最容易出错的地方。

### 3.3 推理工具：模型目录自描述 + 定点参考解释器

以前的 `student-stylize` 有三个毛病：必须传训练 config、必须传 checkpoint、
`--quant` 走的是 burn 的伪量化模拟而**不是真正的导出权重**。
也就是说：交付目录里的 `weights.bin` 谁也没跑过。

这一版重做成一个 `infer` 子命令，靠模型目录自我描述：

| 文件 | 模式 | 内容 |
|---|---|---|
| `model.json` | fp32 / sim | 架构 + 选项（训练时自动写，不用手填） |
| `model.mpk` | fp32 / sim | fp32 权重 |
| `calibration.json` | sim | 激活 scale |
| `manifest.json` + `weights.bin` | int | **定点参考解释器直接读这两个** |

`--mode auto` 的规则：目录里有 `manifest.json` 就走定点，否则走 fp32。
所以 `infer --model deliverables/w4a8/vangogh` 和 `infer --model runs/vangogh` 都能用，
后者还额外能跑 `--mode sim` 做对照。

新写的 `src/intref.rs` 是 manifest 的**逐比特参考实现**：整数卷积、int32 累加、
per-output-channel requant，除了最后一次小数乘法没有任何浮点。它有两个用途：

1. 让交付的 `w8a8/`、`w4a8/` 目录**真的能出图**（之前的 gap 就在这里）；
2. 给 RTL 一个 golden reference——同一张图、同一个 manifest，输出应该逐字节一致。

写它的过程中抓到一个自己埋的 bug：**带 ReLU 的层，requant 的 clamp 范围必须是
`[0, act_qmax]` 而不是 `[-act_qmax, act_qmax]`**。
因为"先 ReLU 再量化"和"量化时 clamp 到 [0, qmax]"在含 0 的对称网格上完全等价
（round 与 max(·,0) 可交换），所以硬件上根本不需要单独的 ReLU。
第一版漏了这一点，输出直接变成噪声（10.6 dB）——修完 49.03 dB。
现在这个范围在 manifest 里用 `out_relu` 显式标出来，不再依赖读代码猜。

## 4. 结果：三个路径

所有导出都是**纯 PTQ**，从同一个 fp32 权重出来，没有第二份需要维护的训练产物。
（为什么不做 QAT 见 §5。）

| 风格 | 变体 | 权重打包 | weights.bin | int vs fp32 | RMSE | RMSE/σ | int vs sim |
|---|---|---|---|---|---|---|---|
| ink | W8A8 | `i8-x1` | 375,824 B | 49.17 dB | 0.89 LSB | 5.3 % | 50.67 dB |
| ink | W4A8 | `i4-x2` | 187,912 B | 38.33 dB | 3.10 LSB | 18.1 % | 49.84 dB |
| vangogh | W8A8 | `i8-x1` | 375,824 B | 39.79 dB | 2.70 LSB | 7.8 % | 43.26 dB |
| vangogh | W4A8 | `i4-x2` | 187,912 B | 31.76 dB | 6.62 LSB | 18.8 % | 41.08 dB |

`int vs sim` 全部在 41 dB 以上，比 `int vs fp32` 高 10 dB 以上——
这是"导出没有 bug"的判据。它一旦掉下来，说明 manifest 和训练代码对同一个图的描述不一致。
这条检查在 v3 之前是不存在的（因为 int 路径根本没跑过）。

## 5. 一个有用的发现：相对误差和风格无关

第一眼看上去梵高比水墨"怕量化"得多：W8A8 是 39.79 dB vs 49.17 dB，差了近 10 dB。
但把误差换算成相对量以后，结论完全反过来：

| | RMSE（1/255 灰阶） | 输出信号标准差 | RMSE/σ |
|---|---|---|---|
| ink W8A8 | 0.89 | 17.3 | 5.3 % |
| vangogh W8A8 | 2.70 | 36.7 | 7.8 % |
| ink W4A8 | 3.10 | 17.3 | 18.1 % |
| vangogh W4A8 | 6.62 | 36.7 | 18.8 % |

**W4A8 上两个风格的相对误差几乎完全相同（18.1% vs 18.8%）。**
梵高的 PSNR 低，是因为它的输出对比度本来就是水墨的两倍，同样的相对误差在绝对尺度上更大。

这条对交付很重要：**不要用绝对 PSNR 跨风格比较**，会得出"梵高模型质量差"的错误结论。
所以 `scripts/evaluate.py` 现在同时输出 RMSE 和 RMSE/σ，`metrics.json` 里都有。

顺带也是一个更一般的方法论：**看量化误差要先归一化到信号本身的尺度**，
否则模型越"敢画"看起来就越差。

## 6. 为什么这一版没有做 QAT

v4 已经实测过：**QAT 的收益和 PTQ 的损失成正比**。这个模型上 W4A8 的 QAT 收益是 0
（41.59 → 41.22 dB），W8A8 只有 +0.9 dB。没有余量可捡，就没必要多维护一条训练路径。
（对照：当时把激活范围故意砍半、制造出真实损失时，QAT 800 步拿到 +3.5 dB。）

## 7. 剩下的问题（按性价比排序）

1. **uint 激活仍未解决**（值整整一位）。v4 §4 记了排查方向：
   先单独关掉输出量化复测，再单独验证无符号激活。
   现在输出量化那一半已经修好并独立验证过了，剩下的变量更少。
2. **风格相似度**。这是学生和教师之间最大的一条差距，而且是**损失函数**的问题，
   不是量化的问题。上 VGG Gram 项是明确的方向，代价是训练变慢、要重新标定、
   而且要重新审视 `teacher-eval/` 的评测口径。
3. **每通道一个权重 scale** 让 2bit 无法使用；分组 scale（group 32，GPTQ/AWQ 那套）
   是让它可用的前提。
4. **`--safety` 是手扫的**，应该换成从激活直方图算 MSE 最优阈值。
5. **推理工具还可以再快**：定点解释器是标量 + rayon 多核，单张 384 图约 1 秒；
   上 SIMD 或者干脆走 GPU 会快一个量级。但它现在的定位是 golden reference，
   慢一点反而更可信。

## 8. 复现

```bash
cd fpga_st_train
cargo build --release
scripts/train.sh ink d0 &
scripts/train.sh vangogh d1 &
wait
scripts/ptq.sh runs/ink    w8a8 && scripts/ptq.sh runs/ink    w4a8
scripts/ptq.sh runs/vangogh w8a8 && scripts/ptq.sh runs/vangogh w4a8
scripts/package.sh
python3 scripts/summary.py
```
