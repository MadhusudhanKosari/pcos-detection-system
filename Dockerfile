FROM python:3.9-slim

# Avoid interactive prompts
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Hugging Face uses port 7860
EXPOSE 7860

# Run Flask app
CMD ["python", "src/app.py"]
