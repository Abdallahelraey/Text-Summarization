from fastapi import FastAPI
import uvicorn
import argparse
import subprocess
import mesop as me
import os
from TextSummarizer.routes.inferance import create_app  
from TextSummarizer.viewers.mesop_text_to_text import TextSummarizerUI
from TextSummarizer.viewers.gradio_view import create_gradio_interface
from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.entity import ViewersConfig
from TextSummarizer.logging import logger


class App:
    def __init__(self, mode, config: ViewersConfig):
        self.config = config
        self.mode = mode
        logger.info(f"App initialized with mode: {self.mode} and configuration: {self.config}")

    def run(self):
        logger.info("Starting the application...")
        try:
            if self.mode == "api":
                self.api_mode()
            elif self.mode == "mesop":
                self.ui_mode()
            elif self.mode == "gradio":
                self.gradio_ui()
            else:
                raise ValueError("Invalid mode selected")
        except Exception as e:
            logger.exception(f"An error occurred while running the app: {e}")
            raise

    def api_mode(self):
        app = create_app()
        uvicorn.run(app, host="127.0.0.1", port=8000)

        logger.info("API mode setup completed.")

    def ui_mode(self):
        logger.info("Starting in UI mode...")
        try:
            subprocess.run(["mesop", self.config.text_to_text_module], check=True)
            logger.info("UI mode subprocess ran successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error while running UI mode subprocess: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in UI mode: {e}")
            raise

    def gradio_ui(self):
        interface = create_gradio_interface()
        interface.launch()
        

if __name__ == "__main__":
    logger.info("Starting the main program...")
    parser = argparse.ArgumentParser(description="Run the app in API or UI mode")
    parser.add_argument("mode", choices=["api", "mesop", "gradio"], help="Choose between API or UI mode")
    args = parser.parse_args()
    logger.info(f"Command-line arguments parsed: {args}")

    try:
        config = ConfigurationManager()
        logger.info("ConfigurationManager initialized.")
        
        viewers_configs = config.get_viwers_config()
        logger.info(f"Viewers configuration retrieved: {viewers_configs}")
        
        app = App(args.mode, viewers_configs)
        app.run()
        logger.info("Application ran successfully.")
    except Exception as e:
        logger.exception(f"An error occurred during application startup: {e}")
        raise
