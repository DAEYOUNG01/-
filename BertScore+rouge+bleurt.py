from bert_score import score
import pandas as pd
import evaluate

# CSV 로드
df = pd.read_csv("/home/elicer/Project/CSV/train.csv")
targets = df["target"].astype(str).apply(lambda x: x[:512]).tolist()
preds = df["source"].astype(str).apply(lambda x: x[:512]).tolist()

# BERTScore
P, R, F1 = score(preds, targets, lang="ko", model_type="skt/kobert-base-v1", num_layers=12)
df["bertscore_precision"] = P.tolist()
df["bertscore_recall"] = R.tolist()
df["bertscore_f1"] = F1.tolist()

# ROUGE
rouge = evaluate.load("rouge")
rouge1_list, rouge2_list, rougeL_list = [], [], []
for pred, tgt in zip(preds, targets):
    r = rouge.compute(predictions=[pred], references=[tgt])
    rouge1_list.append(r["rouge1"])
    rouge2_list.append(r["rouge2"])
    rougeL_list.append(r["rougeL"])

df["rouge1"] = rouge1_list
df["rouge2"] = rouge2_list
df["rougeL"] = rougeL_list

# BLEURT
bleurt = evaluate.load("bleurt", module_type="metric")
bleurt_scores = [bleurt.compute(predictions=[p], references=[t])["scores"][0] for p, t in zip(preds, targets)]
df["bleurt"] = bleurt_scores

# 저장
df.to_csv("/home/elicer/Project/results/train_eval.csv", index=False)
print("✅ 모든 평가 지표 저장 완료!")
