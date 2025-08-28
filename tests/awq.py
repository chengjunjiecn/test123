!pip install autoawq
!pip install transformers
 
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 模型路径（纯文本模型）
model_path = "./qwen"  # 替换为你的模型路径
quant_path = "./quant_awq_gguf"  # 量化后模型保存路径

# 加载模型和分词器
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version":"GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)


# url = "https://drive.google.com/uc?id=14zbm4468eNwWUWcYfmT27ZuoYNF3pQYM&export=download"
# output = "model.gguf"
# gdown.download(url, output, quiet=False)
