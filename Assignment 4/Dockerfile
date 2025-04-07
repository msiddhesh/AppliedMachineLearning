# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY score.py .
COPY best_spam_model.joblib .

# Expose the port the app runs on
EXPOSE 5000

# Command to run on container start
CMD ["python", "app.py"]