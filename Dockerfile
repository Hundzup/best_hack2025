FROM python:3.10
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /hack_best
CMD ["uvicorn", "backend.main:app", "--reload", "--port", "8000", "--host", "0.0.0.0"]