from TextSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline
import os

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_summarizer_config()
        self.tokenizer_name = "FTModel_Tokenzer"
        self.model_name = "FTsummarizer_model"


    
    def predict(self,text):
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.config.output_dir,self.tokenizer_name))
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

        pipe = pipeline("summarization", model=os.path.join(self.config.output_dir,self.model_name),tokenizer=tokenizer)

        print("Meta Text:")
        print(text)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)

        return output