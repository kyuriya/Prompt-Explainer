from transformers import pipeline

# def initialize_model(model_name="gpt2"):
#     """Hugging Face 모델 초기화"""
#     return pipeline("text-generation", model=model_name)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def initialize_model(model_name="Qwen/Qwen2.5-1.5B-Instruct", device=0):
    """모델과 토크나이저 초기화"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(f"cuda:{device}" if device >= 0 else "cpu")
    return model, tokenizer

def get_huggingface_response(model_and_tokenizer, prompt, max_length=100):
    """Hugging Face 모델로부터 응답 생성"""
    try:
        model, tokenizer = model_and_tokenizer
        device = next(model.parameters()).device
        
        # pad_token이 없으면 eos_token을 pad_token으로 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Define the system prompt with the description
        system_prompt = (
            "Generate only the Python code for the given task. Do not include comments or explanations in the code. Ensure the code is complete and functional.\n"
        )
        
        full_prompt = system_prompt + prompt
                
        # 입력을 모델의 디바이스로 이동
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        )
        
        # 모든 텐서를 디바이스로 이동
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # attention_mask 확인 및 생성
        if 'attention_mask' not in inputs or inputs['attention_mask'] is None:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids']).to(device)
        
        print(f"Input device: {inputs['input_ids'].device}")  # 디버깅용
        print(f"Attention mask device: {inputs['attention_mask'].device}")  # 디버깅용
        
        # 응답 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                min_length=20,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                early_stopping=True
            )
            
        # 프롬프트를 제외하고 응답만 디코딩
        prompt_length = len(inputs['input_ids'][0])
        response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        
        print(f"Generated response: {response}")  # 디버깅용
        return response.strip()  # 앞뒤 공백 제거
        
    except Exception as e:
        print(f"Error generating response: {e}")
        import traceback
        print(traceback.format_exc())
        return None
