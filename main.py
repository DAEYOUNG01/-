import pandas as pd
from datasets import Dataset
from dataset import preprocess
from train import train_model
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import numpy as np

# KoBART 토크나이저 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")
model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")

# 512 토큰 단위로 긴 문장 분할하는 함수
def split_long_text(row, max_tokens=512):
    source = row["source"]
    target = row["target"]
    tokens = tokenizer.encode(source)

    if len(tokens) <= max_tokens:
        return [dict(row)]  # ✅ Series → dict로 변환해서 반환

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.decode(tokens[i:i+max_tokens], skip_special_tokens=True)
        chunks.append({"source": chunk, "target": target})

    return chunks


# 전체 데이터프레임에 분할 적용
def expand_dataframe(df):
    expanded = []
    for _, row in df.iterrows():
        expanded.extend(split_long_text(row))
    return pd.DataFrame(expanded)

def main():
    print("✅ main 시작됨")

    train_df = pd.read_csv("/home/elicer/Project/CSV/train.csv")
    val_df = pd.read_csv("/home/elicer/Project/CSV/val.csv")
    print("✅ CSV 로딩 완료")

    train_df = train_df.dropna(subset=["source", "target"]).reset_index(drop=True)
    val_df = val_df.dropna(subset=["source", "target"]).reset_index(drop=True)
    print("✅ 결측치 제거 완료")

    train_df = expand_dataframe(train_df)
    val_df = expand_dataframe(val_df)
    print("✅ 문장 분할 완료")

    val_target_texts = val_df["target"].tolist()
    target_token_lens = [len(tokenizer.encode(text)) for text in val_target_texts]
    avg_target_length = int(np.mean(target_token_lens))
    generation_limit = avg_target_length + 10
    print(f"✅ target 평균 길이: {avg_target_length}, generation_limit: {generation_limit}")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    tokenized_train = train_dataset.map(preprocess, remove_columns=["source", "target"])
    tokenized_val = val_dataset.map(preprocess, remove_columns=["source", "target"])
    print("✅ 토크나이징 완료")

    print("🚀 train_model 호출 시작")
    train_model(tokenized_train, tokenized_val, generation_limit)
    print("🎉 train_model 호출 종료")

if __name__ == "__main__":
    main()

