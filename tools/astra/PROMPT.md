# 从训完的双风格权重里把画面改回来

你直接读、改神经网络的权重。不训练，不用梯度，不换数据集。

两个目标：第二个风格不要再是蓝滤镜；第一个风格的动漫要更分明。方法自己定。下面是已经量过的事实，用来避免再改一段看不见的权重。

## 起点

在仓库根目录工作。

```bash
export PYTHONPATH=tools/astra
export FPGA_ST_TRAIN=$PWD/target/release/fpga_st_train
export FPGA_ST_DEVICE=cpu
```

```python
from weights import Checkpoint
model = Checkpoint.open(
    "experiments/tiny_dw_whitebox_sky_480p/trials/packets"
)
```

这是训完的双风格学生。风格 0 在 `trunk_norm` 和 `film_h2`，是平涂卡通。风格 1 在 `style_packets.0`，输出是罩了蓝灰的照片。照片和两种输出都在这个目录：`content_*.png`、`sample_002000_whitebox_*.png`、`sample_002000_sky_*.png`。

所有写入都是在读出来的数组上改。不要从零写，不要换成单位增益，不要把一个包整个抄到另一个包上再套暖色公式。共享卷积是两个风格一起用的笔，可以小改，但要看到风格 0 还是一张画。

## 已经量过的事

在 `content_landscape.png` 上，只把风格 1 的六块 FiLM 换成风格 0 已经训好的 FiLM，输出就回到卡通：和风格 0 的平均绝对误差从 0.323 降到 0.008，平均色从偏蓝（R−B = −0.15）回到和风格 0 一样。只换主干的 γ 和 β，误差仍是 0.316，蓝色还在。FiLM 的增益路径和偏置路径各换一半，都只恢复一部分（误差 0.14 和 0.09）。

所以蓝滤镜在风格 1 的 `film_h2`，不在主干 γ。上次有人把十来个 γ 乘 1.12，像素差只有 0.001，画面不变。不要再靠这种幅度交差。

`guide.conv.weight` 把原图的边加回全分辨率。猫那张图的光束和脸主要是它。把它乘 0，脸糊成低分辨率色块，边缘幅度从 0.18 掉到 0.06。动漫看起来不够鲜明，是因为这条引导还在贴照片；把它关掉，画会更糊，不会更像动漫。

## 网络

输入输出 RGB，`[0, 1]`。风格包只动两处：主干 LayerNorm 的 γ、β（长度 80），以及 H/2 的 FiLM。

```text
z = relu(reduce(GAP(x)))     # 40 → 10
γ = gamma(z) + 1             # 10 → 40
β = beta(z)                  # 10 → 40
y = x * γ + β
```

γ 的头初始化为 0，所以 `gamma_bias` 为 0 时增益是 1。风格包没有卷积，画不出共享核里没有的笔。斜向笔触不在这些权重里，不要去临摹 `target_002000_sky_*.png`。

## 接口

```python
model.packet_gamma(style)
model.packet_beta(style)
model.set_packet_gamma(style, value)
model.set_packet_beta(style, value)
model.film(style)          # reduce / gamma / beta 的 weight 和 bias
model.set_film(style, parts)
model.get(name)
model.set(name, value)
model.kernels(name)
model.forward(content, out, style=0)
model.save("runs/codex_fix")
```

前向大约 40 次。读权重和已有 png 不算。`style` 只能是 0 或 1。

## 要改成什么样

风格 1 先离开蓝滤镜。杠杆是它自己的六块 FiLM，参照物是风格 0 里已经训好的那套 FiLM。移多少、先动增益还是偏置，由你看权重和前向决定。移完之后，风景应当重新是深底上的那块红布和浅色角，而不是蓝底上的照片。这时它会和风格 0 很像。再从这两套已经训好的数里分开一种用法：铺色更平，或者线条更重。分开之后仍应是画，不是滤镜。

风格 0 要更分明：平坦的地方更平，色块之间对比更清楚，脸和布料不要只是高反差照片。引导核保持在能认脸的范围内。10% 量级的 γ 改动可以当作没看见。改完用猫和风景一起看，不要只看误差。

判断是不是滤镜：逐通道拟合 `a*x+b`。若拟合吃掉绝大部分像素差，而边缘和色块没变，就还是调色，继续改。风格 1 相对照片若仍是同一张图加一层色，也继续改。

## 交回来

保存到 `runs/codex_fix`。写清改了哪些张量、相对加载值的幅度、风景和猫上风格 0 与风格 1 各变成了什么。
