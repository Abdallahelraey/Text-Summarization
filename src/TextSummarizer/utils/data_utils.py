from pymongo import MongoClient
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

class MongoDBHandler:
   def __init__(self, uri: str, db_name: str):
       self.client = MongoClient(uri)
       self.db = self.client[db_name]
       self.texts = self.db['texts']
       self.summaries = self.db['summaries']

   def upload_dataset(self, dataset, batch_size: int = 1000):
       df = dataset.to_pandas() if not isinstance(dataset, pd.DataFrame) else dataset
       records = df.to_dict('records')
       
       for i in range(0, len(records), batch_size):
           self.texts.insert_many(records[i:i + batch_size])

   def save_summary(self, text: str, summary: str, model_name: str) -> str:
       doc = {
           "text": text,
           "summary": summary,
           "model": model_name,
           "created_at": datetime.now()
       }
       result = self.summaries.insert_one(doc)
       return str(result.inserted_id)

   def get_summary(self, text_id: str) -> Dict:
       return self.summaries.find_one({"_id": text_id})

   def get_summaries_by_model(self, model_name: str) -> List[Dict]:
       return list(self.summaries.find({"model": model_name}))

   def close(self):
       self.client.close()

   def __enter__(self):
       return self

   def __exit__(self, exc_type, exc_val, exc_tb):
       self.close()
       
    # Using context manager
    # with MongoDBHandler(uri="mongodb://...", db_name="summarization_db") as db:
    #     # Upload dataset
    #     db.upload_dataset(my_dataset)
        
    #     # Save summary
    #     summary_id = db.save_summary(
    #         text="Original text",
    #         summary="Summarized text",
    #         model_name="t5-base"
    #     )
        
    #     # Get summary
    #     summary = db.get_summary(summary_id)
    



from datasets import load_dataset, Dataset, DatasetDict
from pathlib import Path

class DataUtils:
    def __init__(self, artifacts_path: str):
        self.artifacts_path = Path(artifacts_path)

    def load_data(self, filename: str, file_format: str = "csv") -> Dataset:
        file_path = str(self.artifacts_path / filename)
        return load_dataset(file_format, data_files=file_path)

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
