from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.data_standerization import DataStandardization
from TextSummarizer.logging import logger


class DataStanderizationPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        standardization_config = config.get_data_standardization_config()
        standardizer = DataStandardization(config=standardization_config, remove_stops=True,  lemmatize=True)
        df = standardizer.load_and_prepare_data()
        processed_df = standardizer.process_dataframe(df)
        standardizer.save_data(processed_df)