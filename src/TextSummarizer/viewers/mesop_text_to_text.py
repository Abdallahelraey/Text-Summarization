import mesop as me
import mesop.labs as mel
from TextSummarizer.pipeline.inferance_pipeline import PredictionPipeline

def load(e: me.LoadEvent):
    me.set_theme_mode("system")

class TextSummarizerUI:
    def __init__(self):
        self.prediction_pipeline = PredictionPipeline()
        
    def summarise(self, text: str):
        try:
            result = self.prediction_pipeline.predict(text)
            return result
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            return {"summary": "Error occurred during summarization"}

    def create_page(self):
        # Define the page outside of __init__
        @me.page(
            path="/",
            title="Text Summarizer",
            on_load=load
        )
        def page():
            mel.text_to_text(
                self.summarise,
                title="Natural Language Summarizer"
            )

# Create a single instance of the app
app = TextSummarizerUI()
# Register the page
app.create_page()

