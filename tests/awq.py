!pip install autoawq
!pip install transformers
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 模型路径
model_path = "/content/drive/MyDrive/data/model-f16.gguf"
out_dir = "./quant_awq_gguf"

# 加载 HF 模型或 GGUF FP16
model = AutoAWQForCausalLM.from_pretrained(model_path, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 执行 AWQ 量化
quantized_model = model.quantize(bit=4, use_awq=True)  # 4-bit AWQ

# 保存为 GGUF
quantized_model.save_pretrained(out_dir, save_format="gguf")
