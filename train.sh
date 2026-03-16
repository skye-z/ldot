# 安装 LlamaFactory
cd ~
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt

# 熟悉的小伙伴可以使用这个
# llamafactory-cli train setting.yaml
# 不熟悉的可以使用 WebUI
llamafactory-cli webui