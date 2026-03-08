FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ml_experiment/ ./ml_experiment/
CMD ["python", "ml_experiment/palmer-panguin-decisionTree.py"]
