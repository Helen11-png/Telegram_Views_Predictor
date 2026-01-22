import pandas as pd
import numpy as np
import os

print("="*60)
print("ДИАГНОСТИКА ПРОБЛЕМЫ")
print("="*60)

# 1. Проверка файлов
print("1. Проверка файлов данных:")
files = ['data/doc-1766653704.csv', 'data/doc-1768808876.csv']
for file in files:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024 / 1024
        print(f"   ✓ {file}: {size:.2f} MB")
    else:
        print(f"   ❌ {file}: НЕ НАЙДЕН!")

# 2. Проверка первых строк
print("\n2. Первые строки train данных:")
train_sample = pd.read_csv('data/doc-1766653704.csv', nrows=5)
print(train_sample)

print("\n3. Первые строки test данных:")
test_sample = pd.read_csv('data/doc-1768808876.csv', nrows=5)
print(test_sample)

# 3. Проверка колонок
print("\n4. Колонки train:")
train_cols = pd.read_csv('data/doc-1766653704.csv', nrows=0).columns.tolist()
print(f"   {train_cols}")

print("\n5. Колонки test:")
test_cols = pd.read_csv('data/doc-1768808876.csv', nrows=0).columns.tolist()
print(f"   {test_cols}")

# 4. Проверка библиотек
print("\n6. Версии библиотек:")
import catboost
import sklearn
print(f"   CatBoost: {catboost.__version__}")
print(f"   Pandas: {pd.__version__}")
print(f"   NumPy: {np.__version__}")

# 5. Проверка данных на выбросы
print("\n7. Анализ VIEWS в train:")
train_df = pd.read_csv('data/doc-1766653704.csv', usecols=[' VIEWS'])
train_df.columns = train_df.columns.str.strip()
views = train_df['VIEWS']
print(f"   Min: {views.min()}")
print(f"   Max: {views.max()}")
print(f"   Mean: {views.mean():.0f}")
print(f"   Median: {views.median():.0f}")
print(f"   Std: {views.std():.0f}")

# 6. Проверка CPM
print("\n8. Анализ CPM:")
train_df = pd.read_csv('data/doc-1766653704.csv', usecols=[' CPM'])
train_df.columns = train_df.columns.str.strip()
cpm = train_df['CPM']
print(f"   Min: {cpm.min()}")
print(f"   Max: {cpm.max()}")
print(f"   Mean: {cpm.mean():.2f}")
print(f"   Median: {cpm.median():.2f}")