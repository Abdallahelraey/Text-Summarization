from datasets import load_dataset, Dataset, DatasetDict
from pathlib import Path

class DataUtils:
    def __init__(self, artifacts_path: str):
        self.artifacts_path = Path(artifacts_path)

    def load_data(self, filename: str, file_format: str = "csv") -> Dataset:
        file_path = str(self.artifacts_path / filename)
        return load_dataset(file_format, data_files=file_path)['train']

    def save_data(self, dataset: Dataset, filename: str, file_format: str = "csv"):
        file_path = str(self.artifacts_path / filename)
        dataset.to_csv(file_path) if file_format == "csv" else dataset.to_json(file_path)

    def batch_data(self, dataset: Dataset, batch_size: int) -> Dataset:
        return dataset.map(lambda x: x, batched=True, batch_size=batch_size)

    def tokenize_data(self, dataset: Dataset, tokenizer, column: str = "text", max_length: int = 512) -> Dataset:
        def tokenize_function(examples):
            return tokenizer(examples[column], truncation=True, max_length=max_length)

        return dataset.map(tokenize_function, batched=True)

    def ensure_directory(self):
        if not self.artifacts_path.exists():
            self.artifacts_path.mkdir(parents=True, exist_ok=True)
