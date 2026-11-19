FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# The acronym model is built from acronym_data.csv here, never committed (no weights in git).
RUN python acronym_clf.py train --quiet
CMD ["uvicorn", "bot:app", "--host", "0.0.0.0", "--port", "8765"]
