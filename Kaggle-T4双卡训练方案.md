# Kaggle T4 双卡 LlamaFactory 训练方案

下面这套方案基于 Kaggle Notebook 的 `2x T4` 环境，目标是在 Notebook 内完成 LDOT 的双卡全量微调。

核心约束：

1. T4 不支持 `bf16`，必须切到 `fp16`
2. 双卡训练需要 `torchrun`
3. Kaggle 的输出必须落到 `/kaggle/working/`
4. 训练数据需要先从本仓库导出到 LlamaFactory 的 `data/` 目录

## 1. 安装 LlamaFactory

你已经写了这一段，可以继续沿用：

```python
# ====================== 第一步：拉取 LLaMA Factory 并安装依赖 ======================
%rm -rf /kaggle/working/LlamaFactory
!git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git /kaggle/working/LlamaFactory
%cd /kaggle/working/LlamaFactory
!pip install -e .
!pip install -r requirements/metrics.txt

import torch
if torch.cuda.is_available():
    print("GPU数量:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(
            f"GPU {i}:",
            torch.cuda.get_device_name(i),
            "| 显存:",
            round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
            "GB",
        )
else:
    print("⚠️ 未检测到 GPU")
print("Torch:", torch.__version__)
```

## 2. 拉取本仓库并导出训练资产

```python
# ====================== 第二步：拉取 LDOT 仓库 ======================
%cd /kaggle/working
%rm -rf /kaggle/working/ldot
!git clone https://github.com/skye-z/ldot.git /kaggle/working/ldot
```

这套方案改为让 LlamaFactory 自动从 Hugging Face 下载基础模型。

前提：

1. Kaggle Notebook 的 `Internet` 必须开启
2. `Qwen/Qwen3.5-0.8B-Base` 是公开仓库，一般不需要额外登录
3. 建议把 Hugging Face 缓存显式放到 `/kaggle/working/`

推荐这样设置：

```python
import os

os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/kaggle/working/hf_cache/hub"

MODEL_PATH = "Qwen/Qwen3.5-0.8B-Base"
DATASET_ROOT = "/kaggle/input/datasets/skye98/ldot-dataset"
OUTPUT_DIR = "/kaggle/working/lf_output/ldot_t4x2"
```

如果你在 Kaggle 上遇到 Hugging Face 访问频率限制，再额外执行：

```python
from huggingface_hub import login
login(token="你的_HF_TOKEN")
```

然后把你已经上传到 Kaggle Input 的训练数据和 Kaggle 专用配置导入 LlamaFactory：

```python
# ====================== 第三步：导出训练数据和 Kaggle 配置 ======================
!python /kaggle/working/ldot/scripts/export_kaggle_llamafactory_assets.py \
  --llamafactory-dir /kaggle/working/LlamaFactory \
  --model-path "$MODEL_PATH" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR"
```

## 3. 核对配置

```python
# ====================== 第四步：检查 Kaggle 配置 ======================
!sed -n '1,120p' /kaggle/working/LlamaFactory/setting.kaggle.t4x2.yaml
!python - <<'PY'
import json
from pathlib import Path

dataset_info = json.loads(Path("/kaggle/working/LlamaFactory/data/dataset_info.json").read_text())
print(dataset_info)
print("train samples:", len(json.loads(Path("/kaggle/working/LlamaFactory/data/ldot_train_clean_bilingual_train.json").read_text())))
PY
```

## 4. 启动双卡训练

```python
# ====================== 第五步：2x T4 双卡训练 ======================
%cd /kaggle/working/LlamaFactory
!CUDA_VISIBLE_DEVICES=0,1 \
 FORCE_TORCHRUN=1 \
 NPROC_PER_NODE=2 \
 NNODES=1 \
 MASTER_PORT=29501 \
 llamafactory-cli train setting.kaggle.t4x2.yaml
```

## 5. 导出结果

```python
# ====================== 第六步：查看输出 ======================
!find /kaggle/working/lf_output -maxdepth 3 -type f | sort | tail -n 50
```

如果你想打包结果：

```python
!cd /kaggle/working && zip -r ldot_t4x2_output.zip lf_output
```

## 方案说明

当前 Kaggle 专用配置在：

- `setting.kaggle.t4x2.yaml`

关键参数是：

- `compute_type: fp16`
- `batch_size: 1`
- `gradient_accumulation_steps: 8`
- `cutoff_len: 1024`
- `num_train_epochs: 2.0`
- `learning_rate: 8e-6`

这套参数偏保守，目标是优先保证 `2x T4` 能稳跑。

## 如果 OOM

按这个顺序调整：

1. `cutoff_len: 1024 -> 768`
2. `gradient_accumulation_steps: 8 -> 16`
3. `num_train_epochs: 2.0 -> 1.0`

## 如果你想继续强化

建议在 Kaggle 先跑通稳定版，不要一开始就在 Notebook 环境里用激进参数。

Notebook 环境的意义主要是：

- 验证链路
- 跑基线
- 看 loss / 显存 / 速度

不是追求极限训练。
