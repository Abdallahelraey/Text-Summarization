from TextSummarizer.entity import DataValidationConfig
from TextSummarizer.logging import logger
from datetime import datetime
import pandas as pd
import json
import os


class DataValidation:
   def __init__(self, config: DataValidationConfig):
       self.config = config
        # Clear status file at initialization
       open(self.config.STATUS_FILE, 'w').close()

   def validate_all_files_exist(self) -> dict:
        try:
            all_files = os.listdir(os.path.join("artifacts", "data_ingestion"))
            
            found_files = {required_file: required_file in all_files 
                            for required_file in self.config.ALL_REQUIRED_FILES}
            
            status = {
                "validation_passed": all(found_files.values()),
                "required_files": list(self.config.ALL_REQUIRED_FILES),
                "files_found": found_files,
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"Validated file existence. Status: {status['validation_passed']}")
            with open(self.config.STATUS_FILE, 'a') as f:
                json.dump(status, f, indent=4)
                f.write('\n')
                
            return status
            
        except Exception as e:
            raise e

   def validate_all_required_fields_exist(self) -> dict:
       try:
           found_in_files = {col: [] for col in self.config.required_columns}
           
           for file in self.config.ALL_REQUIRED_FILES:
               file_path = os.path.join(self.config.unzip_dir, file)
               df = pd.read_csv(file_path)
               for col in self.config.required_columns:
                   if col in df.columns:
                       found_in_files[col].append(file)

           missing_columns = {col: found_in_files[col] for col in found_in_files 
                            if not found_in_files[col]}

           status = {
               "validation_passed": len(missing_columns) == 0,
               "required_columns": self.config.required_columns,
               "columns_found_in": found_in_files,
               "missing_columns": missing_columns,
               "timestamp": datetime.now().isoformat()
           }
           logger.info(f"Validated required fields existence. Status: {status['validation_passed']}")
           with open(self.config.STATUS_FILE, 'a') as f:
               json.dump(status, f, indent=4)
               f.write('\n')
           return status

       except Exception as e:
           raise e

   def validate(self) -> bool:
       files_status = self.validate_all_files_exist()
       if not files_status["validation_passed"]:
           return False
           
       fields_status = self.validate_all_required_fields_exist()
       return fields_status["validation_passed"]