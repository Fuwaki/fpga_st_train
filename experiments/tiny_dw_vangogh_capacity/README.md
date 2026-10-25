# tiny_dw_vangogh_capacity

**问题**：在固定 Van Gogh 蒸馏设定下（lr=6e-4、gradient=2），`tiny_dw` 的**宽度 / 残差深度 / 解码通道**谁对观感与 val loss 更敏感？

**锚点**：`configs/dw_study_base.json`（学习率等本轮不动，后续 study 再扫 lr）。

| trial_id | base | res_blocks | mid | fine | 意图 |
|----------|------|------------|-----|------|------|
| b32_rb4 | 32 | 4 | 32 | 16 | 当前默认 |
| b24_rb4 | 24 | 4 | 32 | 16 | 更窄 stem/trunk |
| b40_rb4 | 40 | 4 | 32 | 16 | 更宽 stem/trunk |
| b32_rb3 | 32 | 3 | 32 | 16 | 更浅 trunk |
| b32_rb5 | 32 | 5 | 32 | 16 | 更深 trunk |
| b32_rb6 | 32 | 6 | 32 | 16 | 更深（算力上限探针） |
| b32_dec24_12 | 32 | 4 | 24 | 12 | 更窄解码 |
| b32_dec48_24 | 32 | 4 | 48 | 24 | 更宽解码 |

**数据**：训练 content 走 `data.hf`（`hf-hub` + COCO parquet，`max_shards: 10`）；demo 用 `scripts/fetch_style_transfer_data.sh`。

**一次性准备**（数据、配置、release 编译、HF 试跑）：

```bash
cd fpga_st_train
scripts/prepare_study.sh tiny_dw_vangogh_capacity d1
```

然后：

```bash
scripts/study_run.sh tiny_dw_vangogh_capacity d1
```

若 `experiments/tiny_dw_vangogh_capacity/configs/` 缺失，先生成：

```bash
python scripts/study_gen.py experiments/tiny_dw_vangogh_capacity
```

跑完后：`python scripts/study_collect.py experiments/tiny_dw_vangogh_capacity`，再结合 `trials/*/sample_*.png` 看图。

**教师权重**：默认解析到仓库旁的 `../teacher-eval/MicroAST/models`。若不在该路径：

```bash
export TEACHER_WEIGHTS_DIR=/你的路径/MicroAST/models
scripts/study_run.sh tiny_dw_vangogh_capacity d1
```
