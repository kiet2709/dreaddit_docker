FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

WORKDIR /app

# Copy data + code
COPY data/dreaddit_train.csv /app/data/
COPY data/dreaddit_test.csv /app/data/
COPY code/train.py /app/

# Cài thư viện (KHÔNG CẦN datasets hay parquet)
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    huggingface-hub==0.19.4 \
    transformers==4.36.0 \
    pandas \
    scikit-learn \
    tqdm

CMD ["python", "train.py"]