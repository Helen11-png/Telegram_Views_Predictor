import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Создаем папку data если её нет
os.makedirs('data', exist_ok=True)


def generate_synthetic_data(num_samples=5000, train_format=True):
    np.random.seed(42)
    channel_prefixes = ['tech', 'business', 'news', 'travel', 'entertainment', 'sports', 'education', 'lifestyle']
    channel_names = [f'{prefix}_channel_{i}' for i in range(1, 51) for prefix in channel_prefixes]
    channel_names = random.sample(channel_names, 50)
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=90)
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    data = []
    for i in range(num_samples):
        if np.random.random() < 0.85:
            cpm = round(np.random.uniform(1, 10), 2)
        else:
            cpm = round(np.random.uniform(10, 100), 2)
        base_views = 1000 / (cpm ** 0.7)
        channel_factor = np.random.uniform(0.3, 3.0)
        date = random.choice(dates)
        is_weekend = date.weekday() >= 5
        weekend_factor = 1.2 if is_weekend else 1.0
        views = int(base_views * channel_factor * weekend_factor * np.random.uniform(0.8, 1.2))
        views = max(10, min(2000, views)) 
        if train_format:
            ctr = np.random.beta(2, 20)  # типичный CTR ~2-10%
            clicks = int(views * ctr)
            actions = int(clicks * np.random.beta(1, 10))  # конверсия
            ad_id = 10000 + i
            data.append([ad_id, cpm, views, clicks, actions,
                         random.choice(channel_names), date.strftime('%Y-%m-%d')])
        else:
            # Для тестового формата
            data.append([cpm, random.choice(channel_names),
                         date.strftime('%Y-%m-%d'), np.nan])

    if train_format:
        columns = ['AD_ID', 'CPM', 'VIEWS', 'CLICKS', 'ACTIONS', 'CHANNEL_NAME', 'DATE']
    else:
        columns = ['CPM', 'CHANNEL_NAME', 'DATE', 'VIEWS']

    return pd.DataFrame(data, columns=columns)


if __name__ == "__main__":
    print("Генерация синтетических данных...")

    # Тренировочные данные
    train_synth = generate_synthetic_data(num_samples=5000, train_format=True)
    train_synth.to_csv('data/synthetic_train.csv', index=False)
    print(f"✅ Сгенерирован train: {len(train_synth)} записей")

    # Тестовые данные
    test_synth = generate_synthetic_data(num_samples=3000, train_format=False)
    test_synth.to_csv('data/synthetic_test.csv', index=False)
    print(f"✅ Сгенерирован test: {len(test_synth)} записей")

    # Маленький пример для README
    sample = train_synth.head(20)[['CPM', 'VIEWS', 'CHANNEL_NAME', 'DATE']]
    sample.to_csv('data/sample_data.csv', index=False)
    print("✅ Создан sample_data.csv (20 записей)")

    print("\n🎉 Все данные сгенерированы и сохранены в папке data/")