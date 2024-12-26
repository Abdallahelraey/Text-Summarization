# Use a lightweight Python base image
FROM python:3.11.0

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Set the working directory
WORKDIR /app

# Copy only the requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r deployment_requirements.txt

# Copy the project files
COPY . .

# Install the project in editable mode (separate from requirements.txt)
RUN pip install -e .


# Expose the Gradio port
EXPOSE 7860

# Command to run the app
CMD ["python", "app.py", "gradio"]