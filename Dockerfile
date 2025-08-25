FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
  && pip install -r requirements.txt

COPY api /app/api
COPY scheduled_task /app/scheduled_task

RUN PYTHONPATH=/app pytest api/tests/ -vv

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]