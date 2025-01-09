from transformers import pipeline

def initialize_model(model_name="gpt2"):
    """Hugging Face 모델 초기화"""
    return pipeline("text-generation", model=model_name)

def get_huggingface_response(model, prompt, max_length=150):
    """Hugging Face 모델 호출"""
    response = model(prompt, max_length=max_length, num_return_sequences=1)
    return response[0]["generated_text"]