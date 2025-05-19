# 이 스크립트는 모델 추론 및 평가 결과를 CSV로 저장하며 BLEURT와 Rouge를 활용함

# 라이브러리 임포트
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import pandas as pd
import torch
from tqdm import tqdm
import evaluate
import re

# 메트릭 로드
rouge = evaluate.load("rouge")
bleurt = evaluate.load("bleurt", module_type="metric")

# 모델 및 토크나이저 로드
model_path = "/home/elicer/Project/kobart_correction_model/best"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 데이터 로드
df = pd.read_csv("/home/elicer/Project/CSV/inference_data.csv")

# 🔧 평균 target 길이 계산
target_length = int(df["target"].apply(lambda x: len(tokenizer.encode(str(x)))).mean())
print(f"📏 평균 target 길이: {target_length}")

# 정규화 함수
def normalize_text(text):
    return re.sub(r'\s+', ' ', str(text)).strip() if isinstance(text, str) else ""

# 결과 저장용 리스트
predictions, references, sources = [], [], []
rouge1_scores, rouge2_scores, rougeL_scores, bleurt_scores = [], [], [], []

# 추론 및 평가 시작
for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        source_text, target_text = str(row["source"]), str(row["target"])
        inputs = tokenizer(source_text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=target_length+10, num_beams=5, early_stopping=True, no_repeat_ngram_size=3)

        decoded_output = normalize_text(tokenizer.decode(output[0], skip_special_tokens=True))
        target_text = normalize_text(target_text)

        predictions.append(decoded_output)
        references.append(target_text)
        sources.append(source_text)

        # ROUGE
        try:
            result = rouge.compute(predictions=[decoded_output], references=[target_text], tokenizer=lambda x: x.split(), use_stemmer=False)
            rouge1_scores.append(result.get("rouge1", 0.0))
            rouge2_scores.append(result.get("rouge2", 0.0))
            rougeL_scores.append(result.get("rougeL", 0.0))
        except:
            rouge1_scores.append(0.0)
            rouge2_scores.append(0.0)
            rougeL_scores.append(0.0)

        # BLEURT
        try:
            bleurt_score = bleurt.compute(predictions=[decoded_output], references=[target_text])["scores"][0]
        except:
            bleurt_score = 0.0
        bleurt_scores.append(bleurt_score)

    except Exception as e:
        print(f"샘플 {idx} 처리 실패: {e}")
        predictions.append("")
        references.append(row.get("target", ""))
        sources.append(row.get("source", ""))
        rouge1_scores.append(0.0)
        rouge2_scores.append(0.0)
        rougeL_scores.append(0.0)
        bleurt_scores.append(0.0)

# 결과 저장
result_df = pd.DataFrame({
    "source": sources,
    "target": references,
    "predicted": predictions,
    "rouge1": rouge1_scores,
    "rouge2": rouge2_scores,
    "rougeL": rougeL_scores,
    "bleurt": bleurt_scores
})

result_df.to_csv("/home/elicer/Project/results/inference_results.csv", index=False)
print("\n✅ 결과 저장 완료: inference_results.csv")

# 평균 성능 출력
def avg(lst): return round(sum(lst)/len(lst), 4) if lst else 0
print("\n===== 평균 성능 =====")
print(f"ROUGE-1: {avg(rouge1_scores)}")
print(f"ROUGE-2: {avg(rouge2_scores)}")
print(f"ROUGE-L: {avg(rougeL_scores)}")
print(f"BLEURT: {avg(bleurt_scores)}")
