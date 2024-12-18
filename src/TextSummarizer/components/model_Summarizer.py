import numpy as np
from transformers import AutoTokenizer ,AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import DataCollatorForSeq2Seq
from TextSummarizer.constants import *
from TextSummarizer.utils.file_utils import *
from TextSummarizer.utils.config_utils import *
from TextSummarizer.utils.lib_utils import *
from  TextSummarizer.logging import logger
from TextSummarizer.entity import TrainingConfig


class SummarizationModel:
    def __init__(self, config: TrainingConfig):
        logger.info("Initializing SummarizationModel with provided configuration.")
        self.config = config
        self.data_path = self.config.data_path
        self.output_dir = self.config.output_dir
        self.device = self.config.device
        self.checkpoint = self.config.checkpoint
        self.max_length = self.config.max_length
        self.min_length = self.config.min_length
        self.prefix = self.config.prefix
        logger.info("Loading tokenizer and model from checkpoint: %s", self.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint).to(self.device)
        logger.info("Tokenizer and model loaded successfully.")
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            max_length=self.max_length
        )
        logger.info("Data collator initialized.")
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            weight_decay=self.config.weight_decay,
            save_total_limit=self.config.save_total_limit,
            num_train_epochs=self.config.num_train_epochs,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=self.config.push_to_hub,
        )
        logger.info("Training arguments configured.")

    def load_data_into_DatasetDict(self) -> DatasetDict:
        logger.info("Loading dataset from path: %s", self.data_path)
        loaded_dataset = load_from_disk(self.data_path)
        logger.info("Dataset loaded successfully with %d splits.", len(loaded_dataset))
        return loaded_dataset

    def compute_metrics(self, eval_pred):
        logger.info("Computing evaluation metrics.")
        rouge = evaluate.load("rouge")
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        logger.info("Metrics computed: %s", result)
        return {k: round(v, 4) for k, v in result.items()}

    def model_trainer(self, dataset):
        logger.info("Initializing Seq2SeqTrainer for training.")
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        logger.info("Training started.")
        trainer.train()
        logger.info("Training completed.")
        return trainer

    def save_model(self, model):
        logger.info("Saving model to output directory: %s", self.output_dir)
        model.save_model(self.output_dir)
        logger.info("Model saved successfully.")

    def predict(self, text: str) -> str:
        logger.info("Generating summary for input text.")
        inputs = self.tokenizer(
            self.prefix + text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=self.max_length,
            min_length=self.min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logger.info("Summary generated: %s", summary)
        return summary
