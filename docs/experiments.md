# 实验目录与工具

训练产物体积大、次数多，不适合和「实验设计」混在同一层。约定如下。

## 三层分工

| 层级 | 路径 | 进 Git？ | 作用 |
|------|------|----------|------|
| 设计 | `experiments/<study_id>/` | 是（清单 + 说明 + 生成的小 JSON） | 一组对比实验的定义 |
| 训练图 | `data.hf`（COCO 等 HF 集）或 `data.content_dir` | 缓存于 `HF_HOME` / `~/.cache/huggingface` | 正式训练用 HF，demo 仍用本地 `demo_inputs/` |
| 产物 | `experiments/<study_id>/trials/<trial_id>/` | **否**（`.gitignore`） | 单次训练输出，与现 `runs/<tag>/` 同构 |
| 交付 | `deliverables/` | 是 | 从**选定**的 trial 或再训练结果打包对外 |
| 笔记 | `runs/note/` | 是 | 版本叙事（v1…vN），不存权重 |

历史上的 `runs/ink`、`runs/vangogh` 已退役：交付以 `deliverables/` 为准；新工作一律走 `experiments/`。

## `experiments/<study_id>/` 布局

```
experiments/
  dw_vangogh_lr/                 # study_id：模型_任务_因子（简短、唯一）
    README.md                    # 目的、因子表、结论（几段即可）
    study.json                   # 机器可读：base + trials（见下）
    configs/                     # study_gen.py 生成，可提交
      lr3e-4_g2.json
      lr6e-4_g2.json
    trials/                      # 训练写入，不提交
      lr6e-4_g2/
        config.json              # train 自动快照
        metrics.jsonl
        model.mpk
        sample_*.png
    aggregate/                   # study_collect.py 输出，可选提交 CSV
      summary.csv
```

**trial_id** 命名：`{因子缩写}{取值}` 用下划线连接，例如 `lr6e-4_g2_st1`（lr=6e-4，gradient=2，stroke=1）。避免空格与中文。

**study_id** 命名：`{arch}_{dataset_or_style}_{question}`，例如 `tiny_dw_vangogh_lr`、`tiny_dw_vangogh_stroke`。

每个 trial 目录与旧 `runs/<tag>/` 相同：`fpga_st_train infer --model <trial_dir>` 仍可用。

## `study.json` 格式

```json
{
  "id": "dw_vangogh_lr",
  "base_config": "configs/dw_study_base.json",
  "trials": [
    {
      "id": "lr6e-4_g2",
      "patch": {
        "optim": { "learning_rate": 0.0006 },
        "loss": { "gradient": 2.0 }
      }
    }
  ]
}
```

`study_gen.py` 会：

1. 读取 `base_config`（相对仓库根目录）
2. 深合并 `patch`
3. 写入 `run.name` = trial id，`run.output_dir` = `experiments/<study_id>/trials/<trial_id>`
4. 输出到 `experiments/<study_id>/configs/<trial_id>.json`

## 命令

```bash
# 生成 configs
python scripts/study_gen.py experiments/dw_vangogh_lr

# 顺序跑（日志在 experiments/<study>/log/<trial>.log）
scripts/study_run.sh dw_vangogh_lr d1

# 汇总 metrics.jsonl → aggregate/summary.csv
python scripts/study_collect.py experiments/dw_vangogh_lr
```

单次训练仍可用 `scripts/train.sh <tag>`，但 tag 应对应 `configs/<tag>.json`。**对比实验优先用 study 流程**，避免 `configs/` 里堆几十个手写文件。

## 生成器：Python，不是 Rust

| 用 Python | 用 Rust |
|-----------|---------|
| JSON 合并、读 `metrics.jsonl`、打 CSV/Markdown | `train` / `calibrate` / `export` / 推理 |
| 与现有 `summary.py`、`evaluate.py`、`gallery.py` 一致 | 需要稳定 CLI 与类型，已在 crate 里 |

实验是「编排 + 分析」，改得勤、依赖少，放在 `scripts/*.py` 最合适；不必为 manifest 再拉 serde 进主程序。

## 与 `configs/` 的关系

| 文件 | 用途 |
|------|------|
| `configs/dw_study_base.json` | 某类研究的锚点（教师、数据、默认 arch） |
| `configs/ink.json` / `vangogh.json` | 可选：再训练交付物时的「生产」配置 |
| `experiments/*/configs/*.json` | 由生成器产出，**可复现**某次 trial |

## 日志

- Study 批量：`experiments/<study_id>/log/<trial_id>.log`
- 零散单次：仍可用 `runs/log/<tag>.log`（目录在 `.gitignore` 里，不提交）
