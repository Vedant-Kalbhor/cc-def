FROM python:3.11-slim

# Prevent Python cache files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose Flask/Gunicorn port
EXPOSE 5000

# Run app using Gunicorn
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:5000", "app:app"]
