import torch
from captum.attr import LLMAttribution, FeatureAblation, TextTokenInput
import matplotlib.pyplot as plt
from io import BytesIO
import warnings

# Ignore warnings
warnings.filterwarnings("ignore", ".*past_key_values.*")
warnings.filterwarnings("ignore", ".*Skipping this token.*")

def generate_heatmap(model_and_tokenizer, prompt, response):
    """Generate heatmap using LLMAttribution for prompt attribution."""
    try:
        # Unpack model and tokenizer
        model, tokenizer = model_and_tokenizer
        
        # 모델의 디바이스 확인
        device = next(model.parameters()).device
        print(f"Model device: {device}")  # 디버깅용
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        # 원본 forward 함수 저장
        original_forward = model.forward
        
        # forward 함수 래핑
        def wrapped_forward(*args, **kwargs):
            # 모든 텐서를 디바이스로 이동
            if args:
                args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            if kwargs:
                kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            return original_forward(*args, **kwargs)
        
        # 래핑된 forward 함수 설정
        model.forward = wrapped_forward
        
        # FeatureAblation을 사용하여 LLMAttribution 설정
        fa = FeatureAblation(model)
        llm_attr = LLMAttribution(fa, tokenizer)
        
        # TextTokenInput 생성
        skip_tokens = [1]
        inp = TextTokenInput(
            text=prompt,
            tokenizer=tokenizer,
            baselines=0,
            skip_tokens=skip_tokens
        )
        
        # 모델 입력을 디바이스로 이동
        model_inputs = inp.to_model_input()
        if isinstance(model_inputs, dict):
            model_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in model_inputs.items()}
            inp._model_input = model_inputs
        
        print("Input tensors prepared")  # 디버깅용
        
        # 어트리뷰션 계산
        with torch.no_grad():
            attr_res = llm_attr.attribute(
                inp=inp,
                target=response,
                skip_tokens=skip_tokens,
                n_steps=50  # 어트리뷰션 계산의 정확도 향상
            )
        
        print("Attribution completed successfully")  # 디버깅용
        
        # 원본 forward 함수 복구
        model.forward = original_forward
        
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
        import traceback
        print(traceback.format_exc())
        return None
