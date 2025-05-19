# 불필요한 경고 메시지를 무시하도록 설정
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 필요한 라이브러리 임포트
import torch
from transformers import (
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, EarlyStoppingCallback, TrainerCallback,
    BartForConditionalGeneration, PreTrainedTokenizerFast
)
from utils import compute_metrics  # compute_metrics는 utils.py에서 가져옴
from model import tokenizer, model

# forced eos 제거 및 generation 관련 config 설정
model.config.forced_eos_token_id = 1
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True

# 학습 로그 출력용 커스텀 콜백
class CustomCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        train_loss = next((round(log.get("loss", -1), 4) for log in reversed(state.log_history) if "loss" in log), -1)
        learning_rate = next((round(log.get("learning_rate", 0), 6) for log in reversed(state.log_history) if "learning_rate" in log), 0)
        
        # Rouge 점수가 0일 경우 경고 출력
        if metrics.get("eval_rouge1", 0) == 0:
            print("\n⚠️ 경고: Rouge 점수가 0입니다. 생성 설정과 데이터를 확인하세요.")
        
        log_dict = {
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "val_loss": round(metrics.get("eval_loss", -1), 4),
            "rouge1": round(metrics.get("eval_rouge1", 0), 4),
            "rouge2": round(metrics.get("eval_rouge2", 0), 4),
            "rougeL": round(metrics.get("eval_rougeL", 0), 4),
            "epoch": round(metrics.get("epoch", 0), 2)
        }
        print("\n\U0001F4E2 Epoch Result:")
        print(log_dict)

# 학습 함수
def train_model(tokenized_train, tokenized_val, generation_max_len=256):
    # 학습 파라미터 출력
    print("\n===== 모델 학습 설정 =====")
    print(f"forced_eos_token_id: {model.config.forced_eos_token_id}")
    print(f"eos_token_id: {model.config.eos_token_id}")
    print(f"pad_token_id: {model.config.pad_token_id}")
    print(f"no_repeat_ngram_size: {model.config.no_repeat_ngram_size}")
    print(f"early_stopping: {model.config.early_stopping}")
    
    # 드롭아웃 설정
    model.config.attention_dropout = 0.1
    model.config.dropout = 0.1

    # 데이터 콜레이터 생성
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="max_length", max_length=512)

    # 학습 인수 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rouge1",
        greater_is_better=True,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        fp16=True,
        num_train_epochs=30,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        logging_dir="./logs",
        logging_strategy="epoch",
        report_to="none",
        dataloader_num_workers=4,
        disable_tqdm=False,  # tqdm 활성화로 진행 상황 확인 가능
        
        # 생성 관련 파라미터 (predict_with_generate=True가 중요)
        predict_with_generate=True,
        generation_max_length=generation_max_len,
        generation_num_beams=5
    )

    # 트레이너 생성
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),  # 5 에폭 동안 개선 없으면 조기 종료
            CustomCallback()
        ]
    )

    print("\n===== 학습 시작 =====")
    print(f"학습 데이터셋 크기: {len(tokenized_train)}")
    print(f"검증 데이터셋 크기: {len(tokenized_val)}")
    
    # 학습 시작
    trainer.train()

    # 모델 저장
    os.makedirs("./kobart_correction_model", exist_ok=True)
    print("\n모델 저장 중...")
    
    # 마지막 모델 저장
    trainer.model.save_pretrained("./kobart_correction_model/last")
    tokenizer.save_pretrained("./kobart_correction_model/last")
    print("✅ 마지막 모델 저장 완료: ./kobart_correction_model/last")

    # 최고 성능 모델 저장
    best_ckpt_path = trainer.state.best_model_checkpoint
    if best_ckpt_path:
        print(f"\n✅ 최고 성능 모델 발견: {best_ckpt_path}")
        best_model = BartForConditionalGeneration.from_pretrained(best_ckpt_path)
        best_model.save_pretrained("./kobart_correction_model/best")
        tokenizer.save_pretrained("./kobart_correction_model/best")
        print("✅ 최고 성능 모델 저장 완료: ./kobart_correction_model/best")
    else:
        print("⚠️ 최고 성능 체크포인트를 찾을 수 없습니다!")

# 학습 실행 (메인 코드에서 호출)
if __name__ == "__main__":
    from prepare_data import load_tokenized_datasets
    
    print("데이터셋 로드 중...")
    try:
        tokenized_datasets = load_tokenized_datasets()
        print("✅ 데이터셋 로드 완료")
        train_model(tokenized_datasets["train"], tokenized_datasets["validation"])
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
