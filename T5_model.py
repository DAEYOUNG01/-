from transformers import AutoModelForSeq2SeqLM,AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "wisenut-nlp-team/KoT5", 
    revision="paraphrase",
    token="허깅페이스 토큰"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "wisenut-nlp-team/KoT5", 
    revision="paraphrase",
    token="허깅페이스 토큰"
)



# 모델 저장 함수 (필요 시)
def save_model():
    model.save_pretrained("./koT5_correction_model")
    tokenizer.save_pretrained("./koT5_correction_model")
