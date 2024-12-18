from typing import Set, List
from  TextSummarizer.logging import logger
from TextSummarizer.entity import DataTransformationConfig
from TextSummarizer.utils.lib_utils import *
from transformers import AutoTokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, load_dataset


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        logger.info("Initializing DataTransformation with config")
        self.config = config
        self.checkpoint = self.config.checkpoint
        self.max_length = self.config.max_length
        self.min_length = self.config.min_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.dataset_path = self.config.data_path
        self.sample_size = self.config.sample_size
        self.prefix = self.config.prefix
        logger.info(f"Initialized with checkpoint={self.checkpoint}, max_length={self.max_length}, min_length={self.min_length}, dataset_path={self.dataset_path}, sample_size={self.sample_size}")
 
    def load_data_into_DatasetDict(self, dataset_type: str = "csv") -> DatasetDict:
        logger.info("Loading data into DatasetDict")
        dataset = load_dataset(dataset_type, data_files=self.dataset_path)
        logger.info(f"Loaded dataset with {dataset_type} from {self.dataset_path}")

        # Ensure required columns exist
        required_features = ['transcript_standardized', 'description_standardized', 'title_standardized']
        feature_names = list(dataset["train"].features.keys())
        assert all(col in feature_names for col in required_features), f"Missing required columns: {required_features}"
        logger.info(f"Dataset contains required features: {required_features}")

        # Remove unnecessary columns
        logger.info("Removing unnecessary columns: ['description', 'tags', 'title', 'ratings', 'transcript']")
        dataset = dataset.remove_columns(['description', 'tags', 'title', 'ratings', 'transcript'])

        # Standardize column names
        logger.info("Renaming columns to standardized names")
        dataset = dataset.rename_column("transcript_standardized", "text")
        dataset = dataset.rename_column("description_standardized", "summary")
        dataset = dataset.rename_column("title_standardized", "title")

        # Split the dataset into train/validation/test sets
        logger.info("Splitting dataset into train/validation/test splits")
        train_test_split = dataset["train"].train_test_split(test_size=0.2)
        train_val_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]

        train_val_split = train_val_dataset.train_test_split(test_size=0.1)
        train_dataset = train_val_split["train"]
        validation_dataset = train_val_split["test"]

        dataset = DatasetDict({
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset
        })
        logger.info("Dataset successfully split into train/validation/test")

        return dataset

    def preprocess_function(self, dataset):
        logger.info("Preprocessing dataset")
        if isinstance(dataset, DatasetDict):
            for split in dataset:
                logger.info(f"Preprocessing split: {split}")
                dataset[split] = dataset[split].map(self._preprocess_single_split)
            logger.info("All splits preprocessed")
            return dataset
        else:
            logger.info("Preprocessing a single dataset")
            return self._preprocess_single_split(dataset)

    def _preprocess_single_split(self, batch):
        logger.info("Preprocessing a single split batch")
        if "text" not in batch or "summary" not in batch:
            raise KeyError(f"Keys 'text' or 'summary' not found. Available keys: {list(batch.keys())}")
        
        inputs = [self.prefix + doc for doc in batch["text"]]
        model_inputs = self.tokenizer(inputs, padding=True, max_length=1024, truncation=True)
        labels = self.tokenizer(text_target=batch["summary"], padding=True, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        logger.info("Successfully tokenized inputs and labels for a batch")
        return model_inputs

    def tokenize_dataset(self, dataset):
        logger.info("Tokenizing dataset")
        tokenized_dataset = dataset.map(self.preprocess_function, batched=True)
        logger.info("Dataset successfully tokenized")
        return tokenized_dataset

    def data_sample_loader(self, dataset: Dataset):
        logger.info("Sampling data from dataset")
        sampled_dataset_random = dataset
        for split in dataset:
            logger.info(f"Sampling {self.sample_size} examples from split: {split}")
            sampled_dataset_random[split] = dataset[split].shuffle(seed=42).select(range(self.sample_size))
        logger.info("Successfully loaded sampled dataset")
        return sampled_dataset_random

    def save_dataset(self, dataset: DatasetDict, dataset_name=None):
        dataset_name = dataset_name
        save_dir = self.config.output_dir
        save_path = os.path.join(save_dir, dataset_name)
        if not isinstance(dataset, DatasetDict):
            raise ValueError("Provided dataset is not a DatasetDict object")

        logger.info(f"Saving DatasetDict to directory: {save_path}")

        try:
            dataset.save_to_disk(save_path)
            logger.info(f"Dataset successfully saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save dataset to {save_path}: {e}")
            raise e
        
