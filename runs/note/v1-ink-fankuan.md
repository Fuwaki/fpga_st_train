# v1 — ink (fankuan) 固定风格学生模型

日期 2026-09-13 · 目标：在 500k 参数以内训练一个把 **水墨（范宽《溪山行旅图》）**
风格烤进权重的小模型，输出与 MicroAST 教师"大致相似"，产物将来要能上 FPGA。

---

## 1. 环境与一个必须记住的坑

| 项 | 值 |
|---|---|
| GPU | AMD Radeon RX 580 2048SP（RADV / Mesa 26.1.5），设备名 `d0` |
| 备选 GPU | NVIDIA Tesla P4（`d1`，本期不用） |
| CPU / 内存 | Ryzen 9 5900X 24 线程 / 27 GB |
| 后端 | burn `wgpu`（WGSL）——项目既定策略，见 `docs/backend.md` |

**坑**：DSH 的 bwrap 文件沙箱会给每条 bash 命令挂一个只含 `null/zero/urandom/tty`
的最小 `/dev`（`bwrap --dev /dev`，与文件策略档位无关），因此沙箱内
`vulkaninfo` 报 "Failed to detect any valid GPUs"、`/dev/dri` 不存在。
这不是机器没卡——宿主机上 `lspci` 两张卡都在。会话把文件策略切到
`danger-full-access` 之后（bash 不再经过 bwrap），RX 580 立刻可见。

> 教训：**在本环境里"看不到 GPU"要先怀疑沙箱，不要怀疑硬件**，更不要因此改用 CPU 训练。

---

## 2. 为什么选 ink / fankuan

- 教师 MicroAST 是**任意风格**网络，风格由运行时输入决定；学生要**固定一种**风格，
  所以风格图就是一次训练的全部风格信息。
- `style-transfer/data/styles/ink/` 下有三张：`fankuan.jpg`(1280×2567)、
  `fuchun.jpg`(3840×132)、`mayuan_water.jpg`(1280×839)。
  `fuchun` 是 29:1 长卷，教师侧风格预处理是按短边缩放 → 7447×256，风格编码器代价爆炸
  （`teacher-eval/REPORT.md` §3.2 记录过 SaMam 因此 OOM）。**排除**。
- 选 `fankuan`：`teacher-eval` 已经用它生成过一整套参考输出与指标，便于横向对比；
  且它是立轴而非长卷，短边缩放到 256 后是 256×513，可接受。

**风格分支离线化**：学生训练时风格恒定 ⇒ style encoder + modulator 的输出是常量。
`teacher::Stylizer` 把它算一次并缓存，每步只剩 content encoder + decoder
（教师论文实测 1.82 GMACs @256²）。若每步重算，风格图 513×256 的像素数是训练裁剪的
两倍，教师成本会直接翻倍。

---

## 3. 学生模型：`TinyStudent`（`src/student/tiny.rs`）

固定风格 ⇒ 教师里所有"随风格变化"的东西都坍缩成常量：

- AdaIN 在风格固定时就是**逐通道仿射**（`(x-μx)/σx·σs + μs`，σs/μs 是常数）；
- Modulator 给出的 `w,b` 也是常数。

所以学生**不需要风格输入、不需要调制、不需要任何归一化层**，就是一个普通的
content → stylised 全卷积网络。

### 结构

```
content [3,H,W]
 enc1  Conv3x3(3  -> 32)  + ReLU          H
 enc2  Conv3x3(32 -> 64)  stride 2 + ReLU H/2
 enc3  Conv3x3(64 -> 64)  stride 2 + ReLU H/4
 res   4 × [Conv3x3(64) + ReLU -> Conv3x3(64)]  残差   H/4
 up1   Nearest2x -> Conv3x3(64->32)+ReLU -> Conv1x1(32->32)+ReLU   H/2
 up2   Nearest2x -> Conv3x3(32->16)+ReLU -> Conv1x1(16->16)+ReLU   H
 head  Conv3x3(16 -> 3)                                        H
```

**376,595 参数**（教师 471,987），全部 layer 是 3x3 / 1x1 卷积 + ReLU + 最近邻上采样 +
反射填充，没有转置卷积、空洞、分组卷积或注意力——每一项都能落到行缓冲卷积数据通路上。

### 三个设计决定及依据

1. **去掉 InstanceNorm（旧学生最大的问题）**
   仓库里原 placeholder 学生（16/32/64 + 每个卷积后 IN + 最近邻上采样）在
   `teacher-eval/runs/metrics256.json` 里 `checkerboard` 高达 0.077–0.194，
   而教师只有 ~1e-6 量级。机理：最近邻上采样产生分块常数信号，紧跟其后的 IN
   会把它重新归一化，等于**放大 2px 奈奎斯特分量**。教师自己的归一化（AdaIN）
   只出现在 1/4 分辨率且在上采样**之前**；这一版学生干脆一层归一化都不要。

2. **解码器镜像教师：3x3 卷积在低分辨率、全分辨率通路窄（16 通道）**
   这不是审美问题而是算力问题。全分辨率上的 3x3 卷积代价按像素数平方增长：
   一个 64->32 的 3x3 @256² 就是 1.2 GMACs，比整个 1/4 分辨率主干还贵。
   镜像 MicroAST 的"低分辨率宽、高分辨率窄"后，学生总 MACs 从 ~3.7 GMACs 降到
   ~2.4 GMACs，实测 step 时间明显下降。代价是细节容量变小——教师用同样的
   16 通道全分辨率尾部就能拿到 LapVar 0.0049，先按它的来。

3. **最近邻上采样而不是 bilinear / pixel-shuffle**
   最近邻在 FPGA 上只是走线；且只要后面不接 IN，实测没有棋盘伪影（见结果）。
   `nn.Upsample(nearest)` 也是教师自己的选择。

---

## 4. 损失函数与训练设置

```
loss = 1.0 * L1(student, teacher)  +  w_grad * gradient_L1(student, teacher)  +  w_tv * TV(student)
```

- `gradient_L1`：对图像做前向差分（水平+垂直）后取 L1，**新加**的项（`src/losses/mod.rs`）。
  动机：逐像素 L1 的最优解是条件均值，在笔触/高频目标上会系统性地"抹平"纹理——
  这正是 `teacher-eval` §5 里旧学生 `PSNR 22 dB / LapVar 0.0020`（改动很轻、接近色调偏移）
  的成因。匹配导数可以消掉这个偏差。
- 训练数据：`style-transfer/data/content`（13394 张 ImageNet 图，短边中位数 160）。
- **教师在线**：每步现算伪 GT（学生吃 crop 后的内容，教师对同一个 crop 上色），
  省掉离线生成与磁盘占用；教师前向只占每步一小部分。
- crop 192（内容图短边中位数 160，只需 1.2x 上采样，尽量保留原生细节），batch 4，
  Adam lr 2e-4，常数学习率。

---

## 5. 实验一：gradient 权重该给多少？

三个只有 `loss.gradient` 不同的配置，各 1200 步（约 0.41 s/step）：

| tag | w_grad |
|---|---|
| `exp-l1` | 0（纯 L1 基线） |
| `exp-g2` | 2.0 |
| `exp-g5` | 5.0 |

**判据**：held-out 采样图上 `lapvar / lapvar_t`（笔触能量打到教师的几成）
以及 `psnr_vs_teacher`。60 步的预实验里 L1 掉了 6 倍而 `grad` 一直卡在 0.0237 不动，
说明 w_grad=0.5 时纹理项对梯度几乎没有贡献——所以要往上试。

### 结果（1200 步，`runs/log/exp-*.log`）

笔触能量比 = `lapvar / lapvar_t`（学生打到了教师的几成）：

| 配置 | animal | cat | city | psnr_vs_teacher |
|---|---|---|---|---|
| `exp-l1`（w=0） | **3%** | 2% | 1% | 27.3–28.4 dB |
| `exp-g2`（w=2） | **28%** | 11% | 8% | 27.3–29.2 dB |
| `exp-g5`（w=5） | **28%** | 13% | 8% | 26.9–28.6 dB |

**结论：**

1. **纯 L1 完全学不到纹理**——去掉纹理项后笔触能量只有教师的 1–3%，
   输出是低对比度的平滑色块（`luma 0.25+-0.039`，教师 `0.266+-0.066`，
   内容 `0.416+-0.218`）。这印证了"L1 的最优解是条件均值"这一机理，
   也解释了 `teacher-eval` 里旧学生"改动很轻"的现象。
2. **梯度项是决定性的**：w=2 就把纹理从 3% 拉到 28%（近 10 倍），
   而 w=5 相比 w=2 几乎没有额外收益、PSNR 还略降。**取 w_grad = 2.0。**
3. **棋盘伪影问题解决**：所有配置的 `checker` 都在 1e-7~1e-8 量级，
   低于教师自身的 4e-7~2e-6。去掉"上采样后的 InstanceNorm"达到了预期效果。
4. 但 **28% 仍然不够**，而且 exp-g2 的纹理在 900→1200 步之间停住了。
   与此同时训练 L1 在 1200 步时仍有 0.031（RMSE≈0.043，而目标 std 只有 0.066），
   说明**远没有拟合充分**。

### 两个诊断

- **单图过拟合**（`configs/diag-overfit.json`，600 步只喂 1 张图）：
  L1 从 0.134 降到 0.0189 就压不动了，采样纹理比仍只有 4%。
  → 说明这**不只是数据量不足**，瓶颈在"学生要把纹理从 1/4 分辨率瓶颈里重新合成出来"，
  而输出空间 L1 给这条路径的梯度太弱。
- **教师输出本身的纹理并不高**：fankuan 风格在 384 尺度上 `lapvar_t` 只有 6e-4~8.5e-4，
  而内容图是 1e-3~1.2e-2——教师把图形**平滑**了十几倍。
  所以"像水墨"里**色调/明度映射占了大头**，纹理是相对次要但一眼可见的部分。
  这也意味着不能只看纹理比，`psnr_vs_teacher` 与 `luma/sat` 同样要盯。
