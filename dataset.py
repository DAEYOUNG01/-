import pandas as pd
from datasets import Dataset
from model import tokenizer 
import numpy as np 

def preprocess(example):
    model_input = tokenizer(example["source"], max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["target"], max_length=512, truncation=True, padding="max_length")

    labels_ids = [token_id if token_id != tokenizer.pad_token_id else -100 for token_id in labels["input_ids"]]
    model_input["labels"] = labels_ids
    #model_input["labels"] = np.array(labels_ids, dtype=np.int64)
    return model_input
