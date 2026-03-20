# 安装 LlamaFactory
cd ~
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt

# 训练前先回到本仓库根目录刷新清洗后的双向训练集
# cd /path/to/ldot
# python3 scripts/clean_existing_datasets.py
# python3 scripts/build_eval_datasets.py
#
# 熟悉的小伙伴可以使用这个
# llamafactory-cli train setting.full.stable.yaml
# llamafactory-cli train setting.full.aggressive.yaml
# 不熟悉的可以使用 WebUI
llamafactory-cli webui
