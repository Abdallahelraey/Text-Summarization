import os
import pandas as pd
import re
import nltk
from typing import Set, Optional, List, Union
from  TextSummarizer.logging import logger
from TextSummarizer.entity import DataStandardizationConfig
from TextSummarizer.utils.lib_utils import *

class DataStandardization:
    def __init__(self, 
                 config: DataStandardizationConfig,
                 remove_stops: bool = True,
                 lemmatize: bool = True,
                 custom_stopwords: Optional[Set[str]] = None):
        logger.info("Initializing DataStandardization with config")
        self.config = config
        self.remove_stops = remove_stops
        self.lemmatize = lemmatize
        self.custom_stopwords = custom_stopwords
        self.main_csv_path = os.path.join(self.config.input_file_directory,self.config.ALL_REQUIRED_FILES[0]) 
        self.transcripts_csv_path = os.path.join(self.config.input_file_directory,self.config.ALL_REQUIRED_FILES[1])
        self.text_columns = self.config.text_columns
        logger.debug(f"Set up paths - Main CSV: {self.main_csv_path}, Transcripts CSV: {self.transcripts_csv_path}")
        self.setup_nlp_utilities()
        
    def setup_nlp_utilities(self):
        logger.info("Setting up NLP utilities")
        setup_nltk_environment(self.config.nltk_dir)
        download_nltk_models(self.config.nltk_dir)
        logger.info("NLP utilities setup completed")

    def clean_text(self, text: str) -> str:
        logger.debug("Starting text cleaning process")
        if pd.isna(text):
            logger.warning("Received NA value for text cleaning")
            return ""
            
        # Convert to lowercase and string type
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        logger.debug("Text cleaning completed")
        return text

    def remove_stopwords(self, text: str, custom_stopwords: Optional[Set[str]] = None) -> str:
        logger.debug("Starting stopwords removal")
        if not text:
            logger.warning("Received empty text for stopwords removal")
            return ""
            
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            if custom_stopwords:
                stop_words.update(custom_stopwords)
                logger.debug(f"Added {len(custom_stopwords)} custom stopwords")
        except LookupError:
            logger.info("Downloading stopwords...")
            nltk.download('stopwords')
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
        
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        logger.debug(f"Removed {len(words) - len(filtered_words)} stopwords")
        return ' '.join(filtered_words)

    def lemmatize_text(self, text: str) -> str:
        logger.debug("Starting text lemmatization")
        if not text:
            logger.warning("Received empty text for lemmatization")
            return ""
            
        try:
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
        except LookupError:
            logger.info("Downloading required models for lemmatization...")
            nltk.download('wordnet')
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
        
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        logger.debug("Lemmatization completed")
        return ' '.join(lemmatized_words)

    def standardize_text(self, text: str) -> str:
        logger.debug("Starting text standardization")
        # Clean the text first
        text = self.clean_text(text)
        
        # Remove stopwords if requested
        if self.remove_stops:
            text = self.remove_stopwords(text, self.custom_stopwords)
        
        # Lemmatize if requested
        if self.lemmatize:
            text = self.lemmatize_text(text)
        
        logger.debug("Text standardization completed")
        return text

    def process_dataframe(self, 
                         df: pd.DataFrame, 
                         suffix: str = '_standardized') -> pd.DataFrame:
        logger.info(f"Processing DataFrame with {len(df)} rows")
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()
        
        # Process each text column
        for column in self.text_columns:
            if column not in df.columns:
                logger.error(f"Column '{column}' not found in DataFrame")
                raise ValueError(f"Column '{column}' not found in DataFrame")
                
            # Create new column name
            new_column = f"{column}{suffix}"
            logger.info(f"Processing column: {column} -> {new_column}")
            
            # Apply standardization to the column
            result_df[new_column] = df[column].apply(self.standardize_text)
            logger.debug(f"Completed processing column: {column}")
        
        logger.info("DataFrame processing completed")
        return result_df

    def load_and_prepare_data(self) -> pd.DataFrame:
        logger.info("Starting data loading and preparation")
        # Load main data
        logger.debug(f"Loading main data from {self.main_csv_path}")
        df_main = pd.read_csv(self.main_csv_path)
        logger.info(f"Loaded main data with {len(df_main)} rows")
        
        # Select relevant fields
        relevant_fields = self.config.relevant_fields
        df_main = df_main[relevant_fields]
        logger.debug(f"Selected {len(relevant_fields)} relevant fields")
        
        # Load transcripts
        logger.debug(f"Loading transcripts from {self.transcripts_csv_path}")
        df_transcripts = pd.read_csv(self.transcripts_csv_path)
        logger.info(f"Loaded transcripts with {len(df_transcripts)} rows")
        
        # Merge dataframes
        merging_key = self.config.merging_key
        logger.debug(f"Merging dataframes on key: {merging_key}")
        merged_df = df_main.merge(df_transcripts, on=merging_key).drop(merging_key, axis=1)
        logger.info(f"Merged DataFrame has {len(merged_df)} rows")
        
        return merged_df

    def save_data(self, data):
        standerised_data_path = os.path.join(self.config.output_dir, self.config.output_file)
        logger.info(f"Saving standardized data to {standerised_data_path}")
        try:
            # Save the standardized data
            data.to_csv(standerised_data_path, index=False)
            logger.info("Data successfully saved")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise