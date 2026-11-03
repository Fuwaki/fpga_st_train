# tiny_dw_whitebox_480p_prune90

**问题**：480p 的 `in` / `ln` 检查点按通道分数切到 90% 档之后，同一套损失还能不能把画质训回来？

**锚点**：`experiments/tiny_dw_whitebox_480p` 的 step 8000。宽度 `base/trunk/up_h4/up_h2/h2_guide/guide` 从 `32/96/32/48/16/8` 收到 `16/80/24/40/10/8`。每臂按自己的分数选通道，不共用下标。

微调从切下来的权重开始，Adam 状态不带过来。学习率 1.5e-4 线性收到 1.5e-5，2000 步。损失和 640×480 的帧与原来相同。

先切权重，再训练：

```bash
cargo run --release --no-default-features --features ndarray -- prune \
  --from experiments/tiny_dw_whitebox_480p/trials/ln \
  --out experiments/tiny_dw_whitebox_480p_prune90/init/ln
cargo run --release --no-default-features --features ndarray -- prune \
  --from experiments/tiny_dw_whitebox_480p/trials/in \
  --out experiments/tiny_dw_whitebox_480p_prune90/init/in

python3 scripts/study_gen.py experiments/tiny_dw_whitebox_480p_prune90
GROUP=2 scripts/study_run.sh tiny_dw_whitebox_480p_prune90 d0
```

`prune` 的默认宽度就是这一档。`init/ln` 和 `init/in` 已经按六张 `content_*.png` 切好：45183 参数、1144.6 Mmac 收到 30621 参数、783.5 Mmac。两条臂的通道下标不同。

教师仍走 WebGPU，这台机器跑不了。学生单独一步（640×480，伪标签）在 CPU 上 batch 1 约 4.3 秒、batch 16 约 55 秒、大约 5.5 GiB。按 batch 16 算，2000 步要三十来个小时，所以这一轮不在 CPU 上开训。
