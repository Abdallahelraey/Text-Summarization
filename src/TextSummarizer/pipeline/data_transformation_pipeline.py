from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.data_transofrmation import DataTransformation
from TextSummarizer.logging import logger

class DataTransformationPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        dataset = data_transformation.load_data_into_DatasetDict()
        model_inputs = data_transformation.preprocess_function(dataset)
        tokenized_dataset = data_transformation.tokenize_dataset(model_inputs)
        data_transformation.save_dataset(tokenized_dataset,dataset_name="transformed_dataset")
        sampled_dataset = data_transformation.data_sample_loader(tokenized_dataset)
        data_transformation.save_dataset(sampled_dataset,dataset_name="sampled_dataset")
        
        
if __name__ == "__main__":
    DataTransformationPipeline = DataTransformationPipeline()
    DataTransformationPipeline.main()
