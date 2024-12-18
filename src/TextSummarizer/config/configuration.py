from TextSummarizer.constants import *
from TextSummarizer.utils.file_utils import *
from TextSummarizer.utils.config_utils import *
from TextSummarizer.utils.lib_utils import *
from TextSummarizer.entity import (DataIngestionConfig,DataValidationConfig,DataStandardizationConfig,DataTransformationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
            required_columns=config.required_columns,
            unzip_dir=config.unzip_dir
        )
        
        return data_validation_config
 
    def get_data_standardization_config(self) -> DataStandardizationConfig:
        config = self.config.data_standardization
        create_directories([config.output_dir])
        data_standardization_config = DataStandardizationConfig(
            input_file_directory=config.input_file_directory,
            output_dir = config.output_dir,
            output_file = config.output_file,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
            text_columns  = config.text_columns,
            relevant_fields = config.relevant_fields,
            merging_key = config.merging_key,
            nltk_dir = config.nltk_dir
            #output_file_path=os.path.join(config.output_dir, config.output_file)
   
        )
        return data_standardization_config
    

    
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            max_length = config.max_length,
            min_length = config.min_length,
            output_dir = config.output_dir,
            prefix = config.prefix,
            checkpoint = config.checkpoint,
            sample_size = config.sample_size
        )

        return data_transformation_config