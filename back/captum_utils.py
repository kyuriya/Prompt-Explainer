# import torch
# from captum.attr import LLMAttribution, FeatureAblation, TextTokenInput
# from transformers import AutoTokenizer
# import matplotlib.pyplot as plt
# from io import BytesIO

# def generate_heatmap(model_and_tokenizer, prompt, response):
#     """Generate heatmap using Captum for prompt attribution."""
#     try:
#         # Unpack model and tokenizer
#         model, tokenizer = model_and_tokenizer

#         # Ensure model is on the same device as input tensors
#         device = next(model.parameters()).device  # Get the model's device (e.g., cuda:0 or cpu)

#         # Prepare input for the model and move to the same device
#         model_input = tokenizer(prompt, return_tensors="pt").to(device)  # Move input to the model's device

#         # Use Captum for feature attribution
#         fa = FeatureAblation(model)
#         llm_attr = LLMAttribution(fa, tokenizer)
#         skip_tokens = [tokenizer.pad_token_id]  # Skip padding token

#         # Prepare Captum input and move it to the correct device
#         inp = TextTokenInput(prompt, tokenizer, skip_tokens=skip_tokens).to(device)

#         # Ensure the target response is processed appropriately
#         target_response = tokenizer(response, return_tensors="pt").to(device)

#         # Attribute results
#         attr_res = llm_attr.attribute(inp, target=target_response, skip_tokens=skip_tokens)

#         # Plot the heatmap
#         fig, ax = attr_res.plot_token_attr(show=False)

#         # Save plot to a BytesIO buffer
#         buf = BytesIO()
#         fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
#         buf.seek(0)

#         return buf  # Return buffer for rendering in Streamlit

#     except Exception as e:
#         print(f"Error generating heatmap: {e}")
#         return None
import torch
from captum.attr import LLMAttribution, FeatureAblation, TextTokenInput
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from io import BytesIO

def generate_heatmap(model_and_tokenizer, prompt, response):
    """Generate heatmap using LLMAttribution for prompt attribution."""
    try:
        # Unpack model and tokenizer
        model, tokenizer = model_and_tokenizer
        device = next(model.parameters()).device
        
        # FeatureAblation을 사용하여 LLMAttribution 설정
        fa = FeatureAblation(model)
        llm_attr = LLMAttribution(fa, tokenizer)
        
        # TextTokenInput을 사용하여 입력 생성
        skip_tokens = [tokenizer.pad_token_id] if tokenizer.pad_token_id is not None else []
        inp = TextTokenInput(
            text=prompt,
            tokenizer=tokenizer,
            skip_tokens=skip_tokens
        )
        
        # 입력을 텐서로 변환하고 디바이스로 이동
        input_tensor = inp.to_tensor().to(device)
        
        # response를 토큰화하고 디바이스로 이동
        target_tokens = tokenizer(response, return_tensors="pt")
        target_tokens = {k: v.to(device) for k, v in target_tokens.items()}
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        # 어트리뷰션 계산
        with torch.no_grad():
            attr_res = llm_attr.attribute(
                inp=inp,
                target=target_tokens['input_ids'],
                skip_tokens=skip_tokens,
                internal_batch_size=1  # 배치 크기를 1로 설정
            )
        
        # 시각화
        fig, ax = attr_res.plot_token_attr(show=False)
        
        # 플롯 저장
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        plt.close()

        return buf

    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return None
