FROM python:3.11-slim

WORKDIR /code

# Install system dependencies for LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy EVERYTHING at once
COPY . .

# Install requirements (make sure -e . is GONE from requirements.txt)
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Port for Hugging Face
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]git