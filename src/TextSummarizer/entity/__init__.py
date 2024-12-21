from dataclasses import dataclass
from pathlib import Path
from typing import List
import torch

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
    
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list
    required_columns: List[str]
    unzip_dir: Path
    


@dataclass(frozen=True)
class DataStandardizationConfig:
    input_file_directory: Path
    output_dir: Path
    output_file: Path
    ALL_REQUIRED_FILES: list
    text_columns: list
    relevant_fields: list
    merging_key: str
    nltk_dir: Path
    
    
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path 
    data_path: Path 
    checkpoint: str 
    max_length: int 
    min_length: int 
    output_dir: Path 
    prefix: str 
    sample_size: int 
    
    
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    data_path: Path
    checkpoint: str
    max_length: int
    min_length: int
    output_dir: str
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    weight_decay: float
    save_total_limit: int
    num_train_epochs: int
    prefix: str
    push_to_hub: bool
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
@dataclass(frozen=True)
class ViewersConfig:
    root_dir: Path
    text_to_text_module: Path
    port: int