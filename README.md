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
