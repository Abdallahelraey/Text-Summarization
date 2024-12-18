from TextSummarizer.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from TextSummarizer.pipeline.data_validation_pipeline import DataValidationPipeline
from TextSummarizer.pipeline.data_standerization_pipeline import DataStanderizationPipeline
from TextSummarizer.pipeline.data_transformation_pipeline import DataTransformationPipeline
from TextSummarizer.pipeline.model_summarizer_pipeline import SummarizationModelPipeline
from TextSummarizer.logging import logger


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<") 
   data_ingestion = DataIngestionPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
except Exception as e:
        logger.exception(e)
        raise e
 
     
STAGE_NAME = "Data validation stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<") 
   data_validation = DataValidationPipeline()
   data_validation.main()
   logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
except Exception as e:
        logger.exception(e)
        raise e
     
     
STAGE_NAME = "Data standerization stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<") 
   data_standerization = DataStanderizationPipeline()
   data_standerization.main()
   logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
except Exception as e:
        logger.exception(e)
        raise e
     
     
     
STAGE_NAME = "Data transformation stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<") 
   data_transformation = DataTransformationPipeline()
   data_transformation.main()
   logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
except Exception as e:
        logger.exception(e)
        raise e
     
     
     
STAGE_NAME = "Model Development stage"
try:
   logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<") 
   SummarizationModel = SummarizationModelPipeline()
   SummarizationModel.main()
   logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==============================x")
except Exception as e:
        logger.exception(e)
        raise e