# Use a slim Python image to keep it lightweight
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /api

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and the models folder
# This assumes your local structure is:
# ./app/main.py
# ./models/MLPRegressor.pkl
COPY ./api ./api
COPY ./models ./models

# Change directory to where the script is so the relative path works
WORKDIR /api/api

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
