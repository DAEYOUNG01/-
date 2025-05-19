from bert_score import score
import pandas as pd

# CSV 로드
df = pd.read_csv("/home/elicer/Project/results/inference_results.csv")

# target / predicted 추출 (문자열 + 길이 제한)
targets = df["target"].astype(str).apply(lambda x: x[:512]).tolist()
preds = df["predicted"].astype(str).apply(lambda x: x[:512]).tolist()

# BERTScore 계산
P, R, F1 = score(
    preds,
    targets,
    lang="ko",
    model_type="skt/kobert-base-v1",
    num_layers=12
)

# 결과를 열로 추가
df["bertscore_precision"] = P.tolist()
df["bertscore_recall"] = R.tolist()
df["bertscore_f1"] = F1.tolist()

# 새 파일로 저장
df.to_csv("/home/elicer/Project/results/inference_results_with_bertscore.csv", index=False)
print("✅ BERTScore 결과가 포함된 CSV 저장 완료!")
