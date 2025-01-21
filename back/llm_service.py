from transformers import pipeline

# def initialize_model(model_name="gpt2"):
#     """Hugging Face 모델 초기화"""
#     return pipeline("text-generation", model=model_name)

from transformers import AutoModelForCausalLM, AutoTokenizer

def initialize_model(model_name="Qwen/Qwen2.5-1.5B-Instruct", device=0):
    """모델과 토크나이저 초기화"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(f"cuda:{device}" if device >= 0 else "cpu")
    return model, tokenizer

# def get_huggingface_response(model, prompt, max_length=150):
#     """Hugging Face 모델 호출"""
#     response = model(prompt, max_length=max_length, num_return_sequences=1)
#     return response[0]["generated_text"]
def get_huggingface_response(model_and_tokenizer, prompt, max_length=150):
    """Hugging Face 모델 호출"""
    model, tokenizer = model_and_tokenizer  # 튜플에서 모델과 토크나이저 분리
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # 토크나이즈 후 GPU로 이동
    outputs = model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)  # 생성된 텍스트 디코딩
