import torch
from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

# --- 1. 验证环境核心 ---
print(f"python version: {torch.__version__}")

# 关键检查：Mac 的 GPU 加速是否可用
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 成功！检测到 MPS (Metal) 加速，将使用 Mac GPU 进行训练。")
else:
    device = torch.device("cpu")
    print("⚠️ 警告：未检测到 MPS，将使用 CPU (速度会很慢)。")

print("-" * 30)

# --- 2. 复现之前的代码 ---
print("正在尝试加载模型 (首次运行会下载模型，请耐心等待)...")

model_name_or_path = "bigscience/mt0-large"

# 配置 LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)

try:
    # 加载模型
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    
    # 把模型移动到 Mac 的 GPU 上 (这步很重要！)
    model.to(device)
    
    # 加载 LoRA
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数量，看看省了多少内存
    model.print_trainable_parameters()
    
    print("\n🎉 恭喜！环境配置完美，代码复现成功！")
    
except Exception as e:
    print(f"\n❌ 出错了: {e}")