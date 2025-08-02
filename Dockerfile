# Use an official Python base image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY app.py .

# Set Python path to include the app directory
ENV PYTHONPATH=/app

# Create necessary directories if they don't exist
RUN mkdir -p models data

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the FastAPI app with Uvicorn using proper module path
CMD ["uvicorn", "src.prediction:app", "--host", "0.0.0.0", "--port", "8000"]
