from typing import Optional

import numpy as np
from transformers import HfArgumentParser, TrainingArguments, AutoConfig, AutoTokenizer, \
    AutoModelForSequenceClassification, EvalPrediction, Trainer
from dataclasses import dataclass, field

from datasets import load_dataset
from evaluate import load

import wandb
@dataclass
class DataTrainingArguments:
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    lr: Optional[float] = field(
        default=5e-5,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )

    batch_size: Optional[int] = field(
        default=32,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


if __name__ == '__main__':
    # we are using MRPC datasets.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # if data_args.task_name is not None:
    raw_datasets = load_dataset("nyu-mll/glue", "mrpc")
    # Use the train split for training, validation split for evaluation during training, and test split for prediction
    # remember that the training samples, evaluation samples and test samples are given as command line arguments



    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(model_args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path)
    # Auto model from pre_trained or from AutoModelForSequenceClassification??
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_path, config=config)


    # Tokenize all texts and align the labels with them.
    def tokenize_function(examples, t):
        return tokenizer(examples["sentence1"], examples["sentence2"], max_length=t.model_max_length,
                         truncation=True)


    # raw_datasets = raw_datasets.map(tokenize_function, batched=True)
    raw_datasets = raw_datasets.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )

    train_dataset = raw_datasets["train"]
    val_dataset = raw_datasets["validation"]
    test_dataset = raw_datasets["test"]

    if data_args.max_train_samples > 0:
        train_dataset = raw_datasets["train"].select(range(data_args.max_train_samples))
    if data_args.max_eval_samples > 0:
        val_dataset = raw_datasets["validation"].select(range(data_args.max_eval_samples))
    if  data_args.max_predict_samples > 0:
        test_dataset = raw_datasets["test"].select(range(data_args.max_predict_samples))
    # define metric for evaluation
    metric = load("accuracy") # metric = load("glue", "mrpc")


    # we will implement the compute_metrics function
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)



    # we will now use the trainer class





    training_args.report_to = "wandb"
    training_args.logging_steps = 1
    training_args.logging_strategy = "steps"
    training_args.learning_rate = data_args.lr
    training_args.per_device_train_batch_size = data_args.batch_size

    wandb.login()
    wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="nimer1najar-hebrew-university-of-jerusalem",
        # Set the wandb project where this run will be logged.
        project="nim",
        # Track hyperparameters and run metadata.
        config={
            "max_train_samples": data_args.max_train_samples,
            "max_eval_samples": data_args.max_eval_samples,
            "max_predict_samples": data_args.max_predict_samples,
            "batch_size": data_args.batch_size,
            "lr": data_args.lr,
            "model_name": model_args.model_path,
            "dataset": "glue mrpc",
            "epochs": training_args.num_train_epochs,
            "seed": training_args.seed,
        },
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )


    if training_args.do_train:
        t_result = trainer.train()
        trainer.log_metrics("train", t_result.metrics)
        trainer.save_model()
        # evaluate
        metrics = trainer.evaluate(eval_dataset=val_dataset)
        wandb.log(metrics)
        model.eval()



    if training_args.do_predict:
        training_args.report_to = []
        predictions = trainer.predict(test_dataset=test_dataset).predictions
        predictions = np.argmax(predictions, axis=1)
        metric.compute(predictions=predictions, references=test_dataset["label"])

        with open('predictions.txt', 'w', encoding='utf-8') as f:
            for idx, pred in enumerate(predictions):
                sentence1 = test_dataset[idx]['sentence1']
                sentence2 = test_dataset[idx]['sentence2']
                output_line = f"{sentence1}###{sentence2}###{pred}\n"
                f.write(output_line)
    wandb.finish()


