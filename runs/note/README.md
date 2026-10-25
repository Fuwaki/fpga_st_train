# runs/note — 实验笔记

每个版本一个文件，命名 `vN-<主题>.md`。内容固定回答四件事：

1. **这一版做了什么**（结构 / 损失 / 数据 / 超参）
2. **为什么这样做**（依据是实测数字，不是"感觉"）
3. **结果**（训练日志与采样图里的客观数字）
4. **下一步**（还没验证的假设，以及打算怎么验证）

## 目录约定

**v6 起**：对比实验走 [`docs/experiments.md`](../../docs/experiments.md) 里的 `experiments/<study>/`；
`runs/` 只保留本目录笔记。对外交付仍以 **`deliverables/*/<风格>/`** 为准（与单次 trial 目录同构，均带 `model.json`）。

| 路径 | 内容 |
|---|---|
| `experiments/<study>/study.json` | 一组 trial 的 base + patch |
| `experiments/<study>/configs/*.json` | 生成器产出的训练配置 |
| `experiments/<study>/trials/<trial>/` | 单次训练产物（`metrics.jsonl`、`model.mpk`、采样图等，**不提交 Git**） |
| `configs/<风格>.json` | 可选：再训练/交付用的「生产」配置 |
| `scripts/train.sh` / `study_run.sh` | 训练入口；日志分别在 `runs/log/` 与 `experiments/<study>/log/` |
| `deliverables/fp32\|w8a8\|w4a8/<风格>/` | 交付的三条路径 |
| `deliverables/gallery/` | 可视化对比 |

## 当前交付物（2026-09-13，v5）

两个风格（ink / vangogh）× 三条路径，全部纯 PTQ 从同一个 fp32 权重导出。
v6（2026-10-14）只选定了下一版 `tiny_dw` 的结构，见 `v6-tiny-dw-in-film.md`，还没有进入下面这张表：

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
| `v6-tiny-dw-in-film.md` | depthwise 学生的结构里程碑：容量、InstanceNorm、主干收到 1 块、两个同宽 1×1 换成 FiLM。**还没替换 v5 的交付物** |
| `v7-input-precision.md` | 输入 RGB565/RGB444/灰度 的实测代价；颜色在输出里起什么作用；**两个交付 bug**：`up3.0.dw.in` scale 偏大 10×（改一个数 +4.5 dB）、`vangogh_lut_dw` 整数输出是常数图 |
| `v8-two-plane-input.md` | 把输入从 3 平面砍到 2 平面（Y + 单一色度轴）：折叠是**精确**的、端到端只掉 0.1 dB、`scripts/fold_stem_2ch.py`。色度真实维度是 1（主轴占 86–93%），8 bpp 下 2 平面比 3 平面高 10 dB |
| `v9-guided-overfit.md` | 单图过拟合。4–8 px 卡在 H/8 主干：`film_both` 推到 H/8 后 `e4-8` 从 0.79 掉到 0.22；guided 停在 H/4 后到 0.68 / `e8-16` 0.78。lapvar 比仍是 0.22 对 0.39。零初始化残差门控把笔触抹平 |
| `v10-whitebox-anime.md` | WhiteBox 动漫、COCO 多图、未见图评估。`b_h4`（788 Mmac）26.66 dB，赢过 `film_both`（1569 Mmac）23.86；H/8 的 `c_v2` 最锐但只是局部滤波。多图上 guide 和跳连都被用起来，v9 的「路径没被用」是单图现象。IN `EPS=1e-5` 压住了 guided 主干，v9 塌掉的四臂都在同一区间。下一步以损失为主：flat / edge / contrast |
| `v11-whitebox-stroke.md` | 偏色的根因是默认初始化让 IN 前的信号只剩 1e-8 方差，eps 1e-8 把它放大成噪声；修 `pre_in_gain = sqrt2`，eps 留 1e-5。取图、卷积反向、教师逐张三处提速，一步 8.7 s → 1.9 s，cohort 12 臂。学生的 4–16 px 只做了教师改动的 0.21，`stroke = 8` 抬到 0.34；`edge` 放大的是照片纹理 |
| `v12-whitebox-long.md` | 1000 步太短：同配置 6000 步 4–16 px 0.34 → 0.51，lr 1.5e-3 到 0.64（不加算力），主干 96 0.59，flat 10 + edge 0.5 平坦区 1.25。gate / resid / decoder 加宽没用。描边所有臂停在 0.70，输出形式的上限也只有 0.79。新 `line` 损失（教师边缘加权的带符号梯度 L1）。加大 crop / batch 每像素只快 9–17% |

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
| `python scripts/study_gen.py experiments/<study>` | 由 `study.json` 生成 trial 配置 |
| `scripts/study_run.sh <study> [device]` | 顺序跑一组对比实验 |
| `python scripts/study_collect.py experiments/<study>` | 汇总 `metrics.jsonl` → CSV |
| `scripts/train.sh <tag> [device]` | 单次训练（`configs/<tag>.json`），日志 `runs/log/` |
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
