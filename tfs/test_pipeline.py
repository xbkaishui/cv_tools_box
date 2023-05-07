# test pipeline, please install transformers first
from transformers import pipeline
from loguru import logger
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
import pytest
import torch


def test_pipeline():
    classifier = pipeline("sentiment-analysis")
    res = classifier("I've been waiting for a HuggingFace course my whole life.")
    logger.info(res)

def test_munual_build_pipeline():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    logger.info("tokenizer result {}", inputs)
    model = AutoModel.from_pretrained(checkpoint)
    outputs = model(**inputs)
    logger.info("AutoModel output shape {}", outputs.last_hidden_state.shape)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    outputs = model(**inputs)
    logger.info("AutoModelForSequenceClassification model result {}", outputs)
    logger.info("AutoModelForSequenceClassification shape {}", outputs.logits.shape)
    # post process
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    logger.info("predictions {}", predictions)
    logger.info("model labels {}", model.config.id2label)


    

if __name__ == "__main__":
    # test_pipeline()
    test_munual_build_pipeline()