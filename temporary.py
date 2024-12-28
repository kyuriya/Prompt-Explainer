import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# GPU 사용량 체크 1
initial_gpu_memory = torch.cuda.memory_allocated(0)

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct" # Qwen/Qwen2.5-Coder-3B-Instruct
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPU 사용량 체크 2
gpu_memory_after_model_load = torch.cuda.memory_allocated(0)

# 프롬프트, 메시지
prompt = "write a quick sort algorithm."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# GPU 사용량 체크 3
gpu_memory_before_inference = torch.cuda.memory_allocated(0)

# Inference 실행
generated_ids = model.generate(**model_inputs, max_new_tokens=512)

# GPU 사용량 체크 4
gpu_memory_after_inference = torch.cuda.memory_allocated(0)

# Response, decode
generated_ids_trimmed = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print(f"Response: {response}")
print(f"Initial GPU Memory: {initial_gpu_memory / 1024**2:.2f} MB")
print(f"GPU Memory After Model Load: {gpu_memory_after_model_load / 1024**2:.2f} MB")
print(f"GPU Memory Before Inference: {gpu_memory_before_inference / 1024**2:.2f} MB")
print(f"GPU Memory After Inference: {gpu_memory_after_inference / 1024**2:.2f} MB")
