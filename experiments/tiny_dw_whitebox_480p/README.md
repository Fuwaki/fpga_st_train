# tiny_dw_whitebox_480p

**问题**：部署是 640×480。上一组在 192 方裁上训，H/4 的 InstanceNorm 只看 48×48，部署时看 160×120。把训练帧改成 640×480 之后，同一套长程配方里 InstanceNorm 和 LayerNorm 还差多少？

**锚点**：`j_ref_in` 与 `a_base`（`experiments/tiny_dw_whitebox_ln`）。结构、损失、学习率都不改。原计划 12000 步，lr 从 1.5e-3 线性到 1.5e-4，实际停在 8000。采样长边 640。

`data.frame` 是 640×480。`crop: 192` 在设了 `frame` 时不使用。图比窗口小才按比例放大再裁，否则从原图上随机取这块。

cohort 里一个学生反传完再换下一个，峰值显存是这一批图加上一个学生的激活。学生从 12 降到 2 省的是每步时间。640×480 在 V100-16GB 上实测（两臂同批，步 3–6 的稳态）：

| batch | 峰值显存 | 每步 | 吞吐 |
|---:|---:|---:|---:|
| 8 | 5806 MiB | 1.05 s | 2.34 Mpix/s |
| 16 | 13710 MiB | 1.55 s | 3.17 Mpix/s |
| 24 | 打满 16142 MiB 后 OOM | — | — |

batch 16 是放得下的最大一档，比 batch 8 多大约 35% 的像素吞吐，还剩约 2.6 GB。计划 12000 步，学习率从 1.5e-3 收到 1.5e-4。跑到约 8500 步时，4000 步之后的 500 步均值一直在 0.32–0.36 之间来回，提前停在检查点 8000。`model.mpk` 就是 `step_008000.mpk`。

| trial | 主干 |
|---|---|
| `in` | InstanceNorm，对应 `j_ref_in` |
| `ln` | LayerNorm，对应 `a_base` |

```bash
python3 scripts/study_gen.py experiments/tiny_dw_whitebox_480p
GROUP=2 scripts/study_run.sh tiny_dw_whitebox_480p d0
```
