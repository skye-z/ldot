# LDOT Eval Suite

这套评测集用于验证 LDOT 在目标任务上的真实表现，不混入训练集。

包含 6 个子集：

- `ldot_eval_community_zh_en.json`
- `ldot_eval_community_en_zh.json`
- `ldot_eval_tech_en_zh.json`
- `ldot_eval_tech_zh_en.json`
- `ldot_eval_slang_zh_en.json`
- `ldot_eval_slang_en_zh.json`

建议按以下维度评估：

1. 忠实
   - 是否漏译、错译、增译。
2. 达意
   - 是否自然、通顺、符合目标语言表达习惯。
3. 雅
   - 是否保留原文语气、论坛语境、黑话和情绪色彩。
4. 术语一致性
   - 技术术语、社区称谓、梗表达是否稳定。
5. 格式保留
   - 代码、命令、链接、emoji、标点和段落结构是否保留。

建议额外记录 3 个硬指标：

- 漏译率
- 术语错误率
- 黑话误译率

每条评测样本现在还带有 `difficulty_score` 字段，满分 `100`。

建议理解方式：

- `0-49`: 容易
- `50-69`: 中等
- `70-84`: 困难
- `85-100`: 很难
