from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import os

app = FastAPI(title="Moscow Geocoder API")

# Добавляем middleware для CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели данных для запросов
class GeocodingRequest(BaseModel):
    address: str
    threshold: float = 0.6
    max_results: int = 1

class TextInput(BaseModel):
    text: str

# Импорт геокодера из отдельного файла
from .geocoder import MoscowGeocoder

# Загрузка данных
try:
    # Получаем путь к текущей директории (backend)
    current_dir = Path(__file__).parent
    # Ищем папку filtered_data в родительской директории
    data_path = current_dir.parent / ".." / "filtered_data" / "dataset_1.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"Данные успешно загружены из: {data_path}")
        print(df.head())
    else:
        print(f"Файл данных не найден по пути: {data_path}. Используем тестовые данные.")
    
    # Инициализация геокодера
    geocoder = MoscowGeocoder(df)
    print("Геокодер успешно инициализирован")
except Exception as e:
    print(f"Ошибка при инициализации геокодера: {e}")
    geocoder = None


@app.post("/api/geocode")
async def geocode(request: GeocodingRequest):
    """
    Эндпоинт для геокодирования адресов Москвы
    """
    if geocoder is None:
        raise HTTPException(status_code=500, detail="Геокодер не инициализирован")
    
    try:
        # Принудительно устанавливаем max_results=1 для получения только одного результата
        result = geocoder.combined_geocoding(
            request.address,
            threshold=request.threshold,
            max_results=1
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка геокодирования: {str(e)}")

@app.post("/api/geocode/structured")
async def geocode_structured(request: GeocodingRequest):
    """
    Эндпоинт для геокодирования адресов Москвы с возвратом результата в строго структурированном формате
    
    Возвращает JSON в формате:
    {
      "searched_address": "",
      "objects": [
        {
          "город": "",
          "улица": "",
          "номер_дома": "",
          "номер_корпуса": "",
          "строение": "",
          "lon": 0,
          "lat": 0,
          "score": 0
        }
      ]
    }
    """
    if geocoder is None:
        raise HTTPException(status_code=500, detail="Геокодер не инициализирован")
    
    try:
        # Принудительно устанавливаем max_results=1 для получения только одного результата
        result = geocoder.combined_geocoding(
            request.address,
            threshold=request.threshold,
            max_results=1
        )
        
        # Форматирование результата в требуемом формате
        if not result["objects"]:
            return {
                "searched_address": request.address,
                "objects": []
            }
        
        # Извлекаем первый и единственный объект
        obj = result["objects"][0]
        
        # Формируем ответ в требуемом формате
        structured_result = {
            "searched_address": request.address,
            "objects": [
                {
                    "город": obj.get("город", ""),
                    "улица": obj.get("улица", ""),
                    "номер_дома": obj.get("номер_дома", ""),
                    "номер_корпуса": obj.get("номер_корпуса", ""),
                    "строение": obj.get("строение", ""),
                    "lon": obj.get("lon", 0),
                    "lat": obj.get("lat", 0),
                    "score": obj.get("score", 0)
                }
            ]
        }
        
        return structured_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка геокодирования: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "geocoder_initialized": geocoder is not None,
        "data_count": len(df) if df is not None else 0
    }

# Настройка статических файлов
BASE_DIR = Path(__file__).resolve().parent.parent
frontend_dir = BASE_DIR / "frontend"

# Проверяем наличие папки frontend и создаем ее при необходимости
if not frontend_dir.exists():
    frontend_dir.mkdir(parents=True, exist_ok=True)

# Монтируем статические файлы
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uvicorn backend.main:app --reload --port 8000