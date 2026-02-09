import requests
import json
import subprocess
import time
import sys

def start_server():
    """Запускает сервер в фоновом режиме"""
    try:
        # Запускаем сервер как отдельный процесс
        server_process = subprocess.Popen(
            [sys.executable, "web_app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Ждём немного, чтобы сервер запустился
        time.sleep(3)
        
        return server_process
    except Exception as e:
        print(f"❌ Не удалось запустить сервер: {e}")
        return None

def test_api():
    url = "http://localhost:8000/predict"
    
    test_data = [
        {"cpm": 2.5, "channel": "tech_channel", "date": "2024-01-15"},
        {"cpm": 10.0, "channel": "news_channel", "date": "2024-06-20"},
        {"cpm": 0.8, "channel": "small_channel", "date": "2024-12-01"},
    ]
    
    for data in test_data:
        try:
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {data}")
                print(f"   → Предсказанные просмотры: {result['predicted_views']}")
            else:
                print(f"❌ Ошибка: {response.status_code}")
                print(f"   {response.text}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Сервер не отвечает. Запущен ли web_app.py?")
            print("   Запустите: python web_app.py")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    print("Тестирование API...")
    
    # Проверяем, работает ли сервер
    try:
        health_check = requests.get("http://localhost:8000/health", timeout=5)
        print("✅ Сервер уже запущен")
    except:
        print("⚠️  Сервер не запущен. Запускаем...")
        server = start_server()
        if not server:
            print("❌ Не удалось запустить сервер")
            sys.exit(1)
        time.sleep(5)  # Даём время на запуск
    
    # Тестируем API
    test_api()
    
    # Закрываем сервер, если мы его запускали
    if 'server' in locals():
        server.terminate()
        print("\n✅ Тестирование завершено, сервер остановлен")
