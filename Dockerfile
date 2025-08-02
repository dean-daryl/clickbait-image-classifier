# Use an official Python base image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Set the working directory to src for uvicorn
WORKDIR /app/clickbait-image-classifier/src

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "prediction:app", "--host", "0.0.0.0", "--port", "8000"]
