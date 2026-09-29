# runs/note — 实验笔记

每个版本一个文件，命名 `vN-<主题>.md`。内容固定回答四件事：

1. **这一版做了什么**（结构 / 损失 / 数据 / 超参）
2. **为什么这样做**（依据是实测数字，不是"感觉"）
3. **结果**（训练日志与采样图里的客观数字）
4. **下一步**（还没验证的假设，以及打算怎么验证）

## 目录约定

约定在 v5 定死。**核心是 `runs/<风格>/` 和 `deliverables/*/<风格>/` 同构**，
两者都带 `model.json`，所以 `fpga_st_train infer --model <目录>` 对两者都能直接用。

| 路径 | 内容 |
|---|---|
| `configs/<风格>.json` | 一次训练的全部超参数（唯一需要 config 的地方） |
| `runs/log/<风格>.log` | 该次训练完整 stdout（`scripts/train.sh` 落盘） |
| `runs/<风格>/model.json` | 架构 + 选项 + 元信息（**自我描述**） |
| `runs/<风格>/model.mpk` | 最终 fp32 权重 |
| `runs/<风格>/step_*.mpk` | 周期性检查点 |
| `runs/<风格>/sample_*_*.png` | 学生采样输出 |
| `runs/<风格>/target_*_*.png` | 同一张图上的 MicroAST 教师输出（对照） |
| `runs/<风格>/quant/<bits>/` | 该风格的导出（manifest + weights.bin + calibration + metrics） |
| `deliverables/fp32\|w8a8\|w4a8/<风格>/` | 交付的三条路径 |
| `deliverables/gallery/` | 可视化对比 |

## 当前交付物（2026-09-13，v5）

两个风格（ink / vangogh）× 三条路径，全部纯 PTQ 从同一个 fp32 权重导出：

| 路径 | 内容 |
|---|---|
| `deliverables/fp32/<风格>/model.mpk` | 训练好的 fp32 学生（376,595 参数） |
| `deliverables/w8a8/<风格>/weights.bin` | 375,824 B 对称 int8 权重（一码一字节） |
| `deliverables/w4a8/<风格>/weights.bin` | 187,912 B 对称 int4 权重（`i4-x2`，一字节两码） |
| `deliverables/*/<风格>/manifest.json` | 每层的 scale / bias_int32 / requant / 执行顺序——FPGA 端只读这一个 |
| `deliverables/*/<风格>/calibration.json` | 16 个量化点的激活 scale 与实测 absmax |
| `deliverables/*/<风格>/metrics.json` | 那套权重实测的量化保真度 |
| `deliverables/README.md` | 交付说明（= `docs/deliverables.md`） |

笔记：

| 文件 | 回答的问题 |
|---|---|
| `v1-ink-fankuan.md` | 学生结构怎么定的；为什么纯 L1 学不到纹理；梯度项权重怎么选 |
| `v2-feature-and-lr.md` | 特征蒸馏为什么失败；学习率才是关键旋钮 |
| `v3-int8-qat.md` | int8 方案、PTQ/QAT 结果、什么时候 QAT 才有用、FPGA 交付格式 |
| `v4-low-bit.md` | 4bit/2bit 扫描、QAT 收益、为什么激活比权重敏感、一个没做成的 uint 激活尝试 |
| `v5-two-styles.md` | 目录约定、两个风格、head 输出范围的修复、4bit 打包、定点参考解释器、**为什么相对误差与风格无关** |

> **v3 / v4 里有一部分数字已经过时。** 那两版是在"模拟前向没有建模最后一级输出量化"的
> 代码上测的，absolute 值偏乐观 0.1 dB 量级，而且当时的 head 输出被对称 clamp 到
> `[-127,127]`（= 0.498 以上的高光被削）。交付时以 `v5-two-styles.md` 和
> `deliverables/*/metrics.json` 为准，那里面的数字全部由定点参考解释器产出。

## 画廊

`deliverables/gallery/index.html` —— 浏览器打开。`overview.png` 是两个风格叠在一起的总览，
`ink.png` / `vangogh.png` 是逐风格大图（行 = 内容图，列 = 原图 / MicroAST 教师 /
学生 fp32 / 学生 W8A8 / 学生 W4A8，每格左上角是该图的 PSNR）。
全部由 `scripts/gallery.py` 从 `runs/<风格>/` 生成，不会和笔记里的表对不上。

## 工具

| 命令 | 作用 |
|---|---|
| `scripts/train.sh <风格> [device]` | 训练一个风格，日志落 `runs/log/` |
| `scripts/ptq.sh <模型目录> <w8a8\|w4a8> [safety] [device] [content]` | 标定 + 导出 + 评估 |
| `scripts/package.sh` | 重建 `deliverables/` 与画廊 |
| `scripts/evaluate.py --model <目录> --quant <导出目录>` | 量化保真度（fp32 / sim / int 三方对比） |
| `scripts/summary.py` | 由 `metrics.json` 生成交付表 |
| `scripts/gallery.py` | 生成可视化对比 |
| `fpga_st_train infer --model <目录> --content <图> --out <目录> [--mode auto]` | 推理（部署入口） |

## 指标口径（与 `teacher-eval/` 一致）

| 指标 | 含义 | 期望 |
|---|---|---|
| `lapvar` | 拉普拉斯能量 = 笔触/纹理强度 | 越接近 `lapvar_t`（教师）越好；v5 两个风格都只保留 ~50% |
| `checker` | 2px 奈奎斯特伪影能量 | 越小越好；教师是 ~1e-6 量级 |
| `psnr_vs_teacher` | 与教师输出的 PSNR | 越低 = 改得越狠；教师自身水平参考 `teacher-eval/REPORT.md` §5 |
| `sat` | 平均饱和度 | 与教师同量级 |
| `rmse/LSB` | 量化误差（1/255 灰阶） | 越小越好 |
| `rmse/σ` | 量化误差 / 输出信号标准差 | **跨风格只能比这个**，绝对值会被对比度带偏 |
