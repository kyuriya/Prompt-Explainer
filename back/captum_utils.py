import torch
from captum.attr import LLMAttribution, FeatureAblation, TextTokenInput, TextTemplateInput
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
import os
import json
from datetime import datetime
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import re

# Ignore warnings
warnings.filterwarnings("ignore", ".*past_key_values.*")
warnings.filterwarnings("ignore", ".*Skipping this token.*")

# def generate_heatmap(model_and_tokenizer, prompt, response, cmap="RdBu"):
#     """
#     Generate heatmap using LLMAttribution for prompt attribution.
#     Args:
#         model_and_tokenizer: Tuple of (model, tokenizer)
#         prompt: Input prompt text
#         response: Target response text
#         cmap: Optional colormap for visualization (e.g. 'Reds', 'Blues', 'Greens')
#     """
#     try:
#         # Unpack model and tokenizer
#         model, tokenizer = model_and_tokenizer
        
#         # 모델의 디바이스 확인
#         device = next(model.parameters()).device
#         print(f"Model device: {device}")  # 디버깅용
        
#         # 모델을 평가 모드로 설정
#         model.eval()
        
#         # 원본 forward 함수 저장
#         original_forward = model.forward
        
#         # forward 함수 래핑
#         def wrapped_forward(*args, **kwargs):
#             # 모든 텐서를 디바이스로 이동
#             if args:
#                 args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
#             if kwargs:
#                 kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
#             return original_forward(*args, **kwargs)
        
#         # 래핑된 forward 함수 설정
#         model.forward = wrapped_forward
        
#         # FeatureAblation을 사용하여 LLMAttribution 설정
#         fa = FeatureAblation(model)
#         llm_attr = LLMAttribution(fa, tokenizer)
        
#         # TextTokenInput 생성
#         skip_tokens = [1]
#         inp = TextTokenInput(
#             text=prompt,
#             tokenizer=tokenizer,
#             baselines=0,
#             skip_tokens=skip_tokens
#         )
        
#         # 모델 입력을 디바이스로 이동
#         model_inputs = inp.to_model_input()
#         if isinstance(model_inputs, dict):
#             model_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
#                           for k, v in model_inputs.items()}
#             inp._model_input = model_inputs
        
#         print("Input tensors prepared")  # 디버깅용
        
#         # 어트리뷰션 계산
#         with torch.no_grad():
#             attr_res = llm_attr.attribute(
#                 inp=inp,
#                 target=response,
#                 skip_tokens=skip_tokens,
#                 n_steps=50  # 어트리뷰션 계산의 정확도 향상
#             )
        
#         print("Attribution completed successfully")  # 디버깅용
        
#         # 원본 forward 함수 복구
#         model.forward = original_forward
        
#         # 기본 시각화
#         plt.figure(figsize=(10, 5))  # 여기서 figure 크기 지정
#         fig, ax = attr_res.plot_token_attr(show=False)
        
#         # cmap이 지정된 경우 히트맵의 컬러맵 직접 수정
#         if cmap:
#             # 기존의 히트맵 찾기
#             for im in ax.get_images():
#                 # 새로운 컬러맵으로 업데이트
#                 im.set_cmap(cmap)
        
#         # 플롯 저장
#         buf = BytesIO()
#         fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
#         buf.seek(0)
#         plt.close()

#         return buf

#     except Exception as e:
#         print(f"Error generating heatmap: {e}")
#         import traceback
#         print(traceback.format_exc())
#         return None
def generate_heatmap(model_and_tokenizer, prompt, response):
    """Generate heatmap using LLMAttribution for sentence-level attribution."""
    try:
        # Unpack model and tokenizer
        model, tokenizer = model_and_tokenizer
        
        device = next(model.parameters()).device
        print(f"Model device: {device}")
        
        model.eval()
        
        original_forward = model.forward
        
        def wrapped_forward(*args, **kwargs):
            if args:
                args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            if kwargs:
                kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            return original_forward(*args, **kwargs)
        
        model.forward = wrapped_forward
        
        fa = FeatureAblation(model)
        llm_attr = LLMAttribution(fa, tokenizer)

        # response 문자열이 "."로 시작하면 제거
        if response.startswith("."):
            response = response[1:]
        # 코드블록이 있으면 그 내부만, 없으면 전체를 처리
        match = re.search(r'```python\s+(.*?)\s+```', response, re.DOTALL)
        if match:
            code_block = match.group(1)
            # 주석 제거
            code_block = re.sub(r'#.*', '', code_block)
            # docstring 제거
            code_block = re.sub(r'""".*?"""', '', code_block, flags=re.DOTALL)
            # return code_block.strip()
            response = code_block.strip()
        
        else:
            # 코드블록이 없어도 코드가 있을 수 있으므로, 주석/docstring만 제거 후 반환
            no_comments = re.sub(r'#.*', '', response)
            no_docstrings = re.sub(r'""".*?"""', '', no_comments, flags=re.DOTALL)
            # return no_docstrings.strip()
            # print(f"Generated response: {response}")  # 디버깅용
            # return response.strip()  # 앞뒤 공백 제거
            response = no_docstrings.strip()

        # 문장 단위로 나누기
        marker = f"```"
        if marker in response:
            response_body = response.split(marker, 1)[1]
            if '\n' in response_body:
                response_lines = response_body.split('\n', 1)[1].split('\n')
            else:
                response_lines = response.split("\n")
        else:
            response_lines = response.split("\n")

        response_lines = [line for line in response_lines if "```" not in line]

        line_attributions = {}
        
        # TextTokenInput 사용
        text_input = TextTokenInput(
            text=prompt,
            tokenizer=tokenizer,
            skip_tokens=[0, 1]  # 특수 토큰 스킵
        )
        
        # 모델 입력 생성 후 디바이스로 이동
        model_input = text_input.to_model_input()
        if isinstance(model_input, torch.Tensor):
            model_input = model_input.to(device)
        elif isinstance(model_input, dict):
            model_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in model_input.items()}
        
        print("Input tensors prepared")
        
        # 각 라인별로 어트리뷰션 계산
        for idx, line in enumerate(response_lines):
            if line.strip():  # 빈 줄 제외
                try:
                    with torch.no_grad():
                        attr_res = llm_attr.attribute(
                            inp=text_input,
                            # target=line,
                            target=line.strip(),#추가
                            n_steps=50
                        )
                
                    # CUDA 텐서를 CPU로 이동 후 NumPy 배열로 변환
                    token_attr_cpu = attr_res.token_attr.cpu().numpy()
                
                # 각 라인의 어트리뷰션 결과 저장
                # line_attributions[str(idx)] = {
                #     'line': line,
                #     'attribution': token_attr_cpu.tolist(),
                #     'input_tokens': attr_res.input_tokens
                # }
                    line_attributions[f"line_{idx}"] = {
                        'line': line.strip(),
                        'attribution': token_attr_cpu.tolist(),
                        'input_tokens': attr_res.input_tokens
                    }
                except Exception as e:
                    print(f"Error processing line {idx}: {e}")
                    continue
        # # 시각화
        # fig, axes = plt.subplots(len(line_attributions), 1, figsize=(15, 2 * len(line_attributions)))

        # if len(line_attributions) == 1:
        #     axes = [axes]

        # for idx, (_, data) in enumerate(line_attributions.items()):
        #     # y축 레이블 길이 제한
        #     y_label = data['line'][:50] + "..." if len(data['line']) > 50 else data['line']
        #     sns.heatmap(
        #         np.array(data['attribution']).reshape(1, -1),
        #         xticklabels=data['input_tokens'],  # x축 토큰
        #         yticklabels=[y_label],            # y축 레이블
        #         cmap="RdBu_r",
        #         center=0,
        #         vmin=-1,  # 컬러 스케일 최소값
        #         vmax=1,   # 컬러 스케일 최대값
        #         cbar_kws={"shrink": 0.5},         # 컬러바 크기 조정
        #         ax=axes[idx]
        #     )
        #     # x축 레이블 회전 및 정렬
        #     axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha="right")
        #     axes[idx].set_yticklabels(axes[idx].get_yticklabels(), fontsize=10)
        
        # 데이터 스케일링
        def normalize_attributions(attributions):
            # StandardScaler로 정규화
            scaler = StandardScaler()
            standardized = scaler.fit_transform(attributions.reshape(-1, 1)).reshape(attributions.shape)
            # [-1, 1] 범위로 변환
            min_val = np.min(standardized)
            max_val = np.max(standardized)
            normalized = 2 * (standardized - min_val) / (max_val - min_val) - 1
            return normalized

        def clean_tokens(tokens):
            """특수 기호 제거."""
            # 리스트의 각 요소에 대해 'Ġ'를 제거
            return [token.replace("Ġ", "") for token in tokens]

        # 히트맵에 필요한 데이터 준비
        raw_token_labels = list(line_attributions.values())[0]['input_tokens']
        token_labels = clean_tokens(raw_token_labels)
        attribution_matrix = np.array(
            [np.mean(data['attribution'], axis=0) for data in line_attributions.values()]
        )
        y_labels = [data['line'][:50] + "..." if len(data['line']) > 50 else data['line'] 
                    for data in line_attributions.values()]

        # 정규화된 attribution_matrix
        normalized_attribution_matrix = normalize_attributions(attribution_matrix)
        
        # 시각화
        plt.figure(figsize=(15, len(y_labels) * 0.8))
        
        ax = sns.heatmap(
            normalized_attribution_matrix,
            xticklabels=token_labels,
            yticklabels=y_labels,
            cmap="RdBu_r",
            center=0,
            cbar_kws={"shrink": 0.5},
            vmin=-1,
            vmax=1
        )

        ax.xaxis.tick_top()
        plt.xticks(rotation=45, ha="center")
        plt.yticks(fontsize=10)
        plt.tight_layout()


        # 어트리뷰션 결과를 JSON으로 저장
        # import tempfile
        # import os
        # temp_dir = tempfile.gettempdir()
        # json_path = os.path.join(temp_dir, 'attribution_data.json')
        # with open(json_path, 'w', encoding='utf-8') as f:
        #     json.dump(line_attributions, f, ensure_ascii=False, indent=2)
        
        # # 시각화
        # fig, axes = plt.subplots(len(line_attributions), 1, 
        #                        figsize=(15, 3*len(line_attributions)))
        # if len(line_attributions) == 1:
        #     axes = [axes]
        
        # for idx, (_, data) in enumerate(line_attributions.items()):
        #     sns.heatmap(np.array(data['attribution']).reshape(1, -1),
        #                xticklabels=data['input_tokens'],
        #                yticklabels=[data['line']],
        #                cmap='RdBu_r',
        #                center=0,
        #                ax=axes[idx])
        #     axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 플롯 저장
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        plt.close()

        return buf
        

    except Exception as e:
        print(f"Error generating heatmap: {e}")
        import traceback
        print(traceback.format_exc())
        return None
