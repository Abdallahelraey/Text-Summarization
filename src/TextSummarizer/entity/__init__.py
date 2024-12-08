from dataclasses import dataclass
from pathlib import Path
from typing import List


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