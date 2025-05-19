def compute_metrics(eval_pred):
    import numpy as np
    import evaluate
    import re
    from model import tokenizer

    # ROUGE 메트릭 로드
    rouge = evaluate.load("rouge")
    
    preds, labels = eval_pred
    
    # 예측이 logits인 경우 처리 (일반적으로 predict_with_generate=True라면 이미 토큰 ID)
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # 레이블에서 패딩(-100) 처리
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # 디코딩
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 텍스트 정규화 함수 (inference 코드와 동일하게)
    def normalize_text(text):
        if not isinstance(text, str):
            return ""
        # 공백 정규화
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # 정규화 적용
    decoded_preds = [normalize_text(pred) for pred in decoded_preds]
    decoded_labels = [normalize_text(label) for label in decoded_labels]
    
    # 디버깅: 최대 3개 샘플 출력
    for i in range(min(3, len(decoded_preds))):
        print(f"\n[샘플 {i+1}]")
        print(f"🔍 예측: {decoded_preds[i]}")
        print(f"🎯 정답: {decoded_labels[i]}")
    
    # 명시적 토큰화 방식으로 ROUGE 계산
    rouge_result = rouge.compute(
        predictions=decoded_preds, 
        references=decoded_labels,
        tokenizer=lambda x: x.split(),  # 공백 기준 명시적 토큰화
        use_stemmer=False  # 한국어에는 stemmer가 적합하지 않음
    )
    
    # None 값 처리
    result = {
        "rouge1": rouge_result["rouge1"] if rouge_result["rouge1"] is not None else 0.0,
        "rouge2": rouge_result["rouge2"] if rouge_result["rouge2"] is not None else 0.0,
        "rougeL": rouge_result["rougeL"] if rouge_result["rougeL"] is not None else 0.0
    }
    
    # 정확도는 계산하지 않음 (텍스트 생성에서 정확한 문자열 일치는 의미가 없음)
    # 대신 ROUGE 점수로 대체
    
    return result
