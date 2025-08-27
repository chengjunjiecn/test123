微调qwenvl
google colab
1. 下载工程
!git clone --depth 1 https://github.com/chengjunjiecn/test123.git

%cd test123

!pip install -e ".[torch,metrics]" --no-build-isolation
!pip install swanlab

2.微调qwen-2.5-vl3b模型
!llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft.yaml

3. 测试
!llamafactory-cli webchat examples/inference/qwen2_5vl_lora_sft.yaml

3.合并lora模型
!llamafactory-cli export examples/merge_lora/qwen2_5vl_lora_sft.yaml
