from TextSummarizer.config.configuration import get_settings
import subprocess
from huggingface_hub import HfApi, login
from TextSummarizer.logging import logger
from dataclasses import dataclass
from pathlib import Path, Union
from typing import Optional
from transformers import Trainer


@dataclass
class ModelUploaderConfig:
    model_name: str
    output_dir: Path
    private: bool = False

class HuggingFaceModelUploader:
    def __init__(self, model_name: str, output_dir: Union[str, Path], private: bool = False):
        """Initialize the uploader with direct parameters instead of config object"""
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.private = private
        self.token = get_settings().HUGGINGFACE_ACCESS_TOKEN
        self.api = HfApi()
        self.user_info = None
        self.repo_id = None
        
        if not self.token:
            raise ValueError("HUGGINGFACE_ACCESS_TOKEN is not set in settings")

    def authenticate(self):
        """Authenticate with Hugging Face using both CLI and API"""
        try:
            # First try API login
            login(token=self.token)
            logger.info("Hugging Face API login successful!")
            
            # Then try CLI login
            result = subprocess.run(
                ["huggingface-cli", "login", "--token", self.token],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Hugging Face CLI login successful!")
            
            # Get user info
            try:
                self.user_info = self.api.whoami(token=self.token)
                if not self.user_info:
                    raise ValueError("Failed to get user info - empty response")
                
                logger.info(f"Authenticated as {self.user_info['name']}.")
                # Set repository ID using the explicitly provided model name
                self.repo_id = f"{self.user_info['name']}/{self.model_name}"
                return True
                
            except Exception as e:
                logger.error(f"Failed to get user info: {str(e)}")
                logger.error(f"Token used (first 4 chars): {self.token[:4]}...")
                raise ValueError(f"Failed to get user info: {str(e)}")

        except subprocess.CalledProcessError as error:
            logger.error(f"Hugging Face CLI login failed: {error}")
            logger.error(f"CLI Output: {error.output}")
            raise
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            logger.error(f"Token used (first 4 chars): {self.token[:4]}...")
            raise ValueError(f"Failed to authenticate with Hugging Face: {str(e)}")

    def create_repository(self):
        """Create a new repository on Hugging Face Hub"""
        try:
            # Check if repo exists first
            try:
                existing_repo = self.api.repo_info(repo_id=self.repo_id, token=self.token)
                logger.info(f"Repository '{self.repo_id}' already exists, will upload to existing repo.")
                return True
            except Exception:
                # Repo doesn't exist, create it
                self.api.create_repo(
                    repo_id=self.repo_id,
                    token=self.token,
                    repo_type="model",
                    private=self.private
                )
                logger.info(f"Repository '{self.repo_id}' created successfully.")
                return True
        except Exception as e:
            logger.error(f"Failed to create/verify repository: {str(e)}")
            raise

    def upload_model_files(self):
        """Upload model files from a directory to Hugging Face Hub"""
        try:
            # Verify output directory exists and has content
            if not self.output_dir.exists():
                raise ValueError(f"Output directory {self.output_dir} does not exist")
            
            if not any(self.output_dir.iterdir()):
                raise ValueError(f"Output directory {self.output_dir} is empty")

            # Upload model files
            logger.info(f"Starting upload from {self.output_dir}")
            self.api.upload_folder(
                repo_id=self.repo_id,
                folder_path=str(self.output_dir),
                commit_message="Upload model files",
                token=self.token
            )
            logger.info(f"Model files uploaded successfully to {self.repo_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload model files: {str(e)}")
            raise

    def upload_trainer_model(self, trainer: Trainer):
        """Upload model directly from a HuggingFace Trainer object"""
        try:
            logger.info(f"Starting trainer upload to {self.repo_id}")
            
            # Split repo_id into namespace and model name
            namespace, model_name = self.repo_id.split('/')
            
            # Push to hub using trainer's built-in method
            trainer.push_to_hub(
                model_name=model_name,
                # use_auth_token=self.token,
                commit_message="Upload model via Trainer",
                # hub_model_id=self.repo_id
            )
            logger.info(f"Trainer model uploaded successfully to {self.repo_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload trainer model: {str(e)}")
            raise

    def run(self, trainer: Optional[Trainer] = None):
        """Execute the complete upload process"""
        try:
            logger.info("Starting model upload process...")
            self.authenticate()
            self.create_repository()
            
            if trainer is not None:
                logger.info("Using Trainer object for upload...")
                self.upload_trainer_model(trainer)
            else:
                logger.info("Using directory upload method...")
                self.upload_model_files()
                
            logger.info("Model upload process completed successfully!")
            return True
        except Exception as e:
            logger.error(f"Model upload process failed: {str(e)}")
            raise
