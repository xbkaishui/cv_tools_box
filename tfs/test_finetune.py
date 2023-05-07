from loguru import logger
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def test_train():
    dataset = load_dataset("glue", "mrpc")
    print(dataset)
    raw_train_dataset = dataset["train"]
    print(raw_train_dataset[0])
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_dataset = tokenizer(
        dataset["train"]["sentence1"],
        dataset["train"]["sentence2"],
        padding=True,
        truncation=True,
    )
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    logger.info("tokenized_datasets {}", tokenized_datasets)
    # print(tokenized_dataset)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    samples = tokenized_datasets["train"][:8]
    samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
    print([len(x) for x in samples["input_ids"]])
    
    batch = data_collator(samples)
    print({k: v.shape for k, v in batch.items()})

    # define model
    from transformers import TrainingArguments
    training_args = TrainingArguments("test-trainer")

    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    print(model.config)
    
    # define trainer
    from transformers import Trainer

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

    
if __name__ == "__main__":
    test_train()