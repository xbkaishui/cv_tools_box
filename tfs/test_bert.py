from transformers import pipeline
from loguru import logger
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer


def test_bert_model_load():
    config = BertConfig()
    # model = BertModel(config)
    model = BertModel.from_pretrained("bert-base-cased", mirror='https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models')
    print(config)
    encoded_sequences = [
        [101, 7592, 999, 102],
        [101, 4658, 1012, 102],
        [101, 3835, 999, 102],
    ]   
    model_inputs = torch.tensor(encoded_sequences)
    output = model(model_inputs)
    logger.info("model output {}", output)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased",  mirror='https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models')
    tokens = tokenizer("Using a Transformer network is simple", padding=True, truncation=True, return_tensors="pt")
    logger.info("tokens {}", tokens)
    output = model(**tokens)
    logger.info("model output {}", output)

    



    
if __name__ == "__main__":
    test_bert_model_load()