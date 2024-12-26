from TextSummarizer.pipeline.inferance_pipeline import PredictionPipeline
import gradio as gr

def create_gradio_interface():

    prediction_pipeline = PredictionPipeline()

    def summarize(text: str):
        try:
            result = prediction_pipeline.predict(text)
            return result
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            return "Error occurred during summarization"

    # Create and return the Gradio interface
    return gr.Interface(fn=summarize, inputs="text", outputs="text", title="Summarization Model")
