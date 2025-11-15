from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import os
import sqlite3
from backend.bd_usage import geocode_address

app = FastAPI(title="Geocoder API")

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


# Инициализация пути к базе данных
DB_PATH = Path(__file__).parent.parent / "filtered_data" / "addresses.db"

# Проверяем наличие базы данных
try:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM addresses")
        count = cursor.fetchone()[0]
        print(f"База данных успешно подключена. Количество записей: {count}")
except Exception as e:
    print(f"Предупреждение: не удалось подключиться к базе данных: {e}")
    print("Будет использоваться тестовый режим.")

@app.post("/api/geocode")
async def geocode(request: GeocodingRequest):
    """
    Эндпоинт для геокодирования адресов Москвы с использованием базы данных SQLite
    
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
    try:
        # Проверяем наличие базы данных
        if not os.path.exists(DB_PATH):
            raise HTTPException(status_code=500, detail=f"База данных не найдена по пути: {DB_PATH}")
        
        # Выполняем геокодирование с использованием функции из bd_usage
        result = geocode_address(request.address, db_path=DB_PATH)
        
        # Если результат пустой, возвращаем структуру с пустым массивом объектов
        if not result.get("objects"):
            return {
                "searched_address": request.address,
                "objects": []
            }
        
        # Принудительно ограничиваем результат одним объектом
        if len(result["objects"]) > 1 and request.max_results == 1:
            result["objects"] = [result["objects"][0]]
        
        return result
    except HTTPException:
        raise
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
    try:
        # Проверяем наличие базы данных
        if not os.path.exists(DB_PATH):
            raise HTTPException(status_code=500, detail=f"База данных не найдена по пути: {DB_PATH}")
        
        # Выполняем геокодирование
        result = geocode_address(request.address, db_path=DB_PATH)
        
        # Форматирование результата в требуемом формате
        if not result.get("objects"):
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
                    "город": obj.get("город", "Москва"),
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка геокодирования: {str(e)}")

@app.get("/health")
async def health_check():
    # Проверяем доступность базы данных
    db_available = False
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            db_available = True
    except:
        db_available = False
        
    return {
        "status": "ok",
        "db_available": db_available,
        "db_path": DB_PATH
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