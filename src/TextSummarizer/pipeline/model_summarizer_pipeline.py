from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.model_Summarizer import SummarizationModel
from TextSummarizer.logging import logger

class SummarizationModelPipeline:
    def __init__(self):
        pass
    def main(self):
        config = ConfigurationManager()
        model_summarizer_config = config.get_model_summarizer_config()
        model_summarizer = SummarizationModel(config=model_summarizer_config)
        data_set = model_summarizer.load_data_into_DatasetDict()
        trained_summarizer = model_summarizer.model_trainer(data_set) 
        model_summarizer.save_model(trained_summarizer)
        test = "Imagine a world overwhelmed by information, where sifting through endless articles, reports, and data consumes valuable time and energy. AI summarization offers a powerful solution, employing natural language processing to condense lengthy texts into concise summaries. These systems utilize two primary approaches: extractive summarization, selecting key sentences directly from the source, and abstractive summarization, generating new, more human-like summaries that capture the core meaning. This technology finds diverse applications, from streamlining news consumption and accelerating research analysis to optimizing business workflows by summarizing meetings and customer feedback. While challenges remain in handling complex language, nuanced meanings, and ensuring factual accuracy, ongoing research continually improves the fluency, coherence, and contextual understanding of AI-generated summaries. Ultimately, AI summarization promises to revolutionize how we process information, empowering us to access key insights quickly and efficiently, unlocking the potential of knowledge for a more informed future."
        logger.info(f"Testing the model cababilityy on the following Text: {test}")
        summary = trained_summarizer.predict(test)
        logger.info(summary)
        
        
if __name__ == "__main__":
    DataTransformationPipeline = SummarizationModelPipeline()
    DataTransformationPipeline.main()