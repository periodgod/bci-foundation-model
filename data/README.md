# 数据账本

建议先建立 `datasets.csv`，字段如下：

```csv
dataset_name,task_type,subjects,sessions,channels,sampling_rate,recording_hours,license,labels,montage,unit,quality_notes,path
```

统计口径：

- `recording_hours` 只算有效 EEG 时长。
- 不用重叠窗口虚增小时数。
- subject/session 必须能用于防止数据泄漏。
- 数据单位必须记录为 `volt`、`microvolt` 或原始 ADC。

