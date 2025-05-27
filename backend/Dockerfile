FROM python:3.10-slim

ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_RETRIES=10

WORKDIR /app

# Install OS packages for pandas, numpy, scipy so pip won’t have to compile them
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pandas \
      python3-numpy \
      python3-scipy \
      build-essential \
      libatlas-base-dev \
      gfortran && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install the rest
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        --default-timeout=$PIP_DEFAULT_TIMEOUT \
        --retries=$PIP_RETRIES \
        -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
