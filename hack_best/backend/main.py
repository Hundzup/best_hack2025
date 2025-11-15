from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

app = FastAPI()

# Модель данных для запроса
class TextInput(BaseModel):
    text: str
    
# Эндпоинт для ML-обработки
@app.post("/api/predict")
async def predict(input_data: TextInput):
    # ЗАГЛУШКА ML-МОДЕЛИ: инвертируем строку
    # prediction = your_ml_model.predict(input_data.text)
    prediction = 'all good!!!'
    print(prediction)

    return {"prediction": prediction}

# Монтируем статические файлы
BASE_DIR = Path(__file__).resolve().parent.parent

app.mount("/", StaticFiles(directory=BASE_DIR / "frontend", html=True), name="static")

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon"}