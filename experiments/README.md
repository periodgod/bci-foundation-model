# Experiments

每次实验建立独立目录：

```text
YYYYMMDD_HHMMSS_experiment_name/
  config.yaml
  splits.json
  run_log.txt
  environment.txt
  checkpoints/
  results/
  figures/
```

实验目录只保存配置、日志、轻量结果和必要 checkpoint。大规模缓存和原始数据不要放入 git。

