# test pipeline, please install transformers first
from transformers import pipeline
from loguru import logger

def test_pipeline():
    classifier = pipeline("sentiment-analysis")
    res = classifier("I've been waiting for a HuggingFace course my whole life.")
    logger.info(res)


if __name__ == "__main__":
    test_pipeline()