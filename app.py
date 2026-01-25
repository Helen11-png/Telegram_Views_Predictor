from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import json
from catboost import CatBoostRegressor
from typing import Optional
import uvicorn
from contextlib import asynccontextmanager

# Модели запросов
class PredictionRequest(BaseModel):
    cpm: float
    channel: str
    date: str  # формат "YYYY-MM-DD"

class PredictionResponse(BaseModel):
    predicted_views: int
    confidence_score: Optional[float] = None

# Глобальные переменные для моделей
model = None
group_stats = []
model_info = {}

# Загрузка моделей при старте
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, group_stats, model_info
    
    try:
        # Загружаем CatBoost модель
        model = CatBoostRegressor()
        model.load_model('models/catboost_model.cbm')
        
        # Загружаем group_stats
        with open('models/group_stats.json', 'r') as f:
            group_stats = json.load(f)
        
        # Загружаем model_info
        with open('models/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        print("✅ Модели успешно загружены")
    except Exception as e:
        print(f"❌ Ошибка загрузки моделей: {e}")
        print("Убедитесь что файлы в папке models/")
        print("Запустите сначала: python model_training.py")
        raise
    
    yield
    
    # Shutdown (при необходимости)
    print("Сервер остановлен")

# Создаём приложение с lifespan
app = FastAPI(
    title="Telegram Ads Predictor API",
    description="API для прогнозирования просмотров рекламы в Telegram",
    version="1.0.0",
    lifespan=lifespan
)

# Вспомогательные функции
def prepare_features(cpm: float, channel: str, date_str: str):
    """Подготовка признаков для модели"""
    try:
        # Преобразуем дату
        date = pd.to_datetime(date_str)
        
        # Создаём DataFrame с одной строкой
        df = pd.DataFrame({
            'CPM': [cpm],
            'CHANNEL_NAME': [channel],
            'DATE': [date]
        })
        
        # Добавляем фичи как в обучении
        df['log_cpm'] = np.log1p(df['CPM'])
        
        # Циклические признаки даты
        df['month'] = date.month
        df['day_of_week'] = date.dayofweek
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    except Exception as e:
        raise ValueError(f"Ошибка подготовки признаков: {e}")

def smart_postprocessing(predictions, cpms, group_stats):
    """Умная постобработка на основе анализа ошибок в валидации"""
    corrected = predictions.copy()
    
    for i in range(len(corrected)):
        cpm = cpms.iloc[i] if hasattr(cpms, 'iloc') else cpms[i]
        
        # Находим к какой группе относится этот CPM
        for stats in group_stats:
            cpm_min = stats.get('cpm_min', 0)
            cpm_max = stats.get('cpm_max', float('inf'))
            
            if cpm_max == float('inf'):
                if cpm >= cpm_min:
                    error_pct = stats.get('error_pct', 0)
                    if error_pct > 5:
                        correction_factor = 1.0 - (error_pct / 100)
                        corrected[i] *= max(correction_factor, 0.7)
                    elif error_pct < -5:
                        correction_factor = 1.0 - (error_pct / 100)
                        corrected[i] *= min(correction_factor, 1.3)
                    break
            else:
                if cpm_min <= cpm < cpm_max:
                    error_pct = stats.get('error_pct', 0)
                    if error_pct > 5:
                        correction_factor = 1.0 - (error_pct / 100)
                        corrected[i] *= max(correction_factor, 0.7)
                    elif error_pct < -5:
                        correction_factor = 1.0 - (error_pct / 100)
                        corrected[i] *= min(correction_factor, 1.3)
                    break
    
    # Физические ограничения
    for i in range(len(corrected)):
        cpm = cpms.iloc[i] if hasattr(cpms, 'iloc') else cpms[i]
        
        if cpm < 0.5:
            corrected[i] = min(corrected[i], 1500)
        elif cpm < 1:
            corrected[i] = min(corrected[i], 1200)
        elif cpm < 2:
            corrected[i] = min(corrected[i], 900)
        elif cpm > 50:
            corrected[i] = max(corrected[i], 30)
        elif cpm > 100:
            corrected[i] = max(corrected[i], 15)
    
    return np.round(corrected).clip(10, 2000).astype(int)

# Основной endpoint - ДОЛЖЕН БЫТЬ POST!
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Прогнозирование количества просмотров для рекламы в Telegram
    
    Параметры:
    - cpm: стоимость за 1000 показов (float)
    - channel: название канала (str)
    - date: дата размещения в формате YYYY-MM-DD (str)
    
    Возвращает:
    - predicted_views: прогнозируемое количество просмотров
    """
    try:
        # Проверяем что модель загружена
        if model is None:
            raise HTTPException(status_code=503, detail="Модель не загружена")
        
        # 1. Подготавливаем данные
        df = prepare_features(request.cpm, request.channel, request.date)
        
        # 2. Делаем предсказание
        features = model_info.get('features', [])
        cat_features = model_info.get('cat_features', [])
        
        if not features:
            raise HTTPException(status_code=500, detail="Ошибка конфигурации модели")
        
        pred_log = model.predict(df[features + cat_features])
        pred_raw = np.expm1(pred_log)
        
        # 3. Применяем постобработку
        final_pred = smart_postprocessing(
            pred_raw, 
            df['CPM'], 
            group_stats
        )[0]
        
        # 4. Возвращаем результат
        return PredictionResponse(
            predicted_views=int(final_pred),
            confidence_score=None
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )

# Health check endpoint - GET!
@app.get("/health")
async def health_check():
    """Проверка работоспособности API"""
    return {
        "status": "healthy",
        "service": "Telegram Ads Predictor",
        "version": "1.0.0",
        "model_loaded": model is not None
    }

# Пример использования - GET!
@app.get("/example")
async def get_example():
    """Пример запроса для тестирования"""
    return {
        "example_request": {
            "cpm": 5.0,
            "channel": "example_channel",
            "date": "2024-01-15"
        },
        "curl_command": """curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"cpm": 5.0, "channel": "test_channel", "date": "2024-01-15"}'""",
        "python_example": """import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"cpm": 5.0, "channel": "test_channel", "date": "2024-01-15"}
)
print(response.json())"""
    }

# Тестовый endpoint для проверки POST
@app.get("/test-predict-get")
async def test_predict_get():
    """Только для тестирования - не использовать в продакшене"""
    try:
        # Имитируем POST запрос
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        response = client.post("/predict", json={
            "cpm": 5.0,
            "channel": "test_channel",
            "date": "2024-01-15"
        })
        
        return {
            "test_result": "GET endpoint для теста",
            "actual_post_should_return": "Используйте POST /predict",
            "example_response": response.json() if response.status_code == 200 else {"error": response.text}
        }
    except:
        return {"message": "Используйте POST запрос на /predict"}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",  
        host="0.0.0.0",
        port=8000,
        reload=True
    )
