from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# 모델 & 토크나이저 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")
model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")

# 저장 (처음에 한 번만 실행하면 됨. 그 이후엔 inference.py에서 다시 로드)
def save_model():
    model.save_pretrained("./kobart_correction_model")
    tokenizer.save_pretrained("./kobart_correction_model")

