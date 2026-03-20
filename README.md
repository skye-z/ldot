# LDOT - Linux Do Translation Model

> 以下内容由AI生成 :)

## 简介

LDOT 专为 [Linux Do](https://linux.do) 社区打造，是一款专注于中英文互译的翻译模型。基于 Qwen3.5 大语言模型微调，能够准确、自然地进行中英文翻译。

## 模型信息

| 属性 | 值 |
|------|-----|
| 基础模型 | Qwen3.5-0.8B |
| 训练方式 | 全量微调 + 知识蒸馏 |
| 参数量 | 约 0.8B |
| 支持语言 | 中文 ↔ 英文 |

## 训练数据

LDOT 基于精选的中英文对照数据进行训练，包括：
- Linux Do 社区讨论
- 技术文档
- 日常对话
- 网络热梗

## 效果示例

| 原文 | 翻译 |
|------|------|
| 啊？多大个脸蛋子呀 | Huh? Who does he think he is? |
| 前排吃瓜中。。。 | Front row watching the drama unfold... |

## 技术栈

- **基础模型**: Qwen3.5
- **微调框架**: LlamaFactory
- **推理引擎**: Ollama / llama.cpp

## 训练配置

仓库当前提供两套全量微调配置：

- `setting.full.stable.yaml`
- `setting.full.aggressive.yaml`

默认的 `setting.yaml` 当前与稳定版保持一致，方便兼容原有训练入口。

当前默认训练数据集为 `dataset/ldot_train_clean_bilingual_train.json`，它由现有数据集清洗、去重、扩展为中英双向翻译语料后生成，并额外纳入了黑话与社区语境一致性专项数据；同时已经剔除了评测集样本，避免训练与验收泄漏。

目录约定：

- `dataset/` 只保留最终训练集和评测集。
- `temp/source/` 存放源语料。
- `temp/` 其余位置存放采集原件、清洗中间结果和训练集构建中间文件。

生成方式：

```bash
python3 scripts/clean_existing_datasets.py
python3 scripts/build_eval_datasets.py
```

两套配置的区别：

| 配置 | 用途 | 主要参数特点 |
|------|------|-------------|
| `setting.full.stable.yaml` | 先追求稳定、低翻车率 | `learning_rate=1e-5`、`warmup_steps=100`、`num_train_epochs=3` |
| `setting.full.aggressive.yaml` | 更强地向 Linux Do 翻译域注入风格和黑话 | `learning_rate=2e-5`、`warmup_steps=20`、`num_train_epochs=4` |

建议理解：

- `stable` 更适合先打基线，观察整体翻译质量、遗忘情况和术语稳定性。
- `aggressive` 更适合在数据质量足够高时强化社区风格、黑话和目标域表达。
- `aggressive` 更容易打出目标域峰值，但也更容易放大脏数据、错译和灾难性遗忘。

推荐训练顺序：

1. 先用 `setting.full.stable.yaml` 跑一版稳定基线。
2. 再用 `setting.full.aggressive.yaml` 跑一版强化版。
3. 用同一套 Linux Do 中英互译评测集对比两版结果，再决定后续默认配置。

训练前建议先刷新清洗后的双向训练集，并重新生成独立评测集：

```bash
python3 scripts/clean_existing_datasets.py
python3 scripts/build_eval_datasets.py
```

评测集会输出到 `dataset/eval/` 目录，训练集会同步更新为 `dataset/ldot_train_clean_bilingual_train.json`；中间文件会输出到 `temp/`。

直接训练：

```bash
llamafactory-cli train setting.full.stable.yaml
llamafactory-cli train setting.full.aggressive.yaml
```

## 贡献

你可以将你认为模型需要知道的或者需要学习的翻译添加到`dataset`目录中，如果你不懂如何创建数据集可以直接在`dataset\ldot_translation.json`中追加，完成后提交 PR 即可！

或者你可以直接提交 Issue

## 许可证

MIT License

## 团队

- **主要开发者**: SkyeZhang
- **开发团队**: BetaX Dev Team

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen) - Base model
- [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) - Fine-tuning framework
- [Linux Do](https://linux.do) - Community support

---

<p align="center">Made with ❤️ by <a href="https://github.com/skye-z">SkyeZhang</a> & <a href="https://betax.dev">BetaX Dev Team</a></p>

<p align="center">
  <img src="https://img.shields.io/badge/HuggingFace-BetaXDev%2FLDOT-blue" alt="HuggingFace">
  <img src="https://img.shields.io/badge/ModelScope-skyezhang%2Fldot-purple" alt="ModelScope">
</p>

<p align="center">
  <a href="https://github.com/skye-z/ldot">GitHub</a> •
  <a href="https://www.modelscope.cn/models/skyezhang/ldot">ModelScope</a> •
  <a href="https://huggingface.co/BetaXDev/LDOT">HuggingFace</a> •
  <a href="https://linux.do">Linux Do</a>
</p>
