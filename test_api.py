import requests
import json

# Тест API
def test_api():
    url = "http://localhost:8000/predict"
    
    test_data = [
        {"cpm": 2.5, "channel": "tech_channel", "date": "2024-01-15"},
        {"cpm": 10.0, "channel": "news_channel", "date": "2024-06-20"},
        {"cpm": 0.8, "channel": "small_channel", "date": "2024-12-01"},
    ]
    
    for data in test_data:
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {data}")
                print(f"   → Предсказанные просмотры: {result['predicted_views']}")
            else:
                print(f"❌ Ошибка: {response.status_code}")
                print(f"   {response.text}")
        except Exception as e:
            print(f"❌ Ошибка соединения: {e}")

if __name__ == "__main__":
    print("Тестирование API...")
    test_api()