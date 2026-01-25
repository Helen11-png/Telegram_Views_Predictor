import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import warnings
import pickle
import json
import os
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)

# ========== ЗАГРУЗКА ==========
train_df = pd.read_csv(
    'data/doc-1766653704.csv',
    usecols=[' CPM', ' VIEWS', ' CHANNEL_NAME', ' DATE']
)

test_df = pd.read_csv('data/doc-1769013311.csv')

# Чистим названия колонок
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# ========== ТОЛЬКО ОЧИСТКА ЭКСТРЕМАЛЬНЫХ ВЫБРОСОВ ==========
# Удаляем только 0.1% самых экстремальных значений
q_low = train_df['VIEWS'].quantile(0.0005)
q_high = train_df['VIEWS'].quantile(0.9995)
train_df = train_df[(train_df['VIEWS'] >= q_low) & (train_df['VIEWS'] <= q_high)]

print(f"После очистки: {train_df.shape}")

# ========== ВАШИ ПРИЗНАКИ (без изменений) ==========
def add_date_features(df):
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['month'] = df['DATE'].dt.month
    df['day_of_week'] = df['DATE'].dt.dayofweek

    # циклические признаки
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df

train_df = add_date_features(train_df)
test_df = add_date_features(test_df)

# ========== ВАШИ ПРЕОБРАЗОВАНИЯ (без изменений) ==========
train_df['log_cpm'] = np.log1p(train_df['CPM'])
test_df['log_cpm'] = np.log1p(test_df['CPM'])
train_df['log_views'] = np.log1p(train_df['VIEWS'])

# ========== РАЗДЕЛЕНИЕ (немного больше валидации) ==========
train_df = train_df.sort_values('DATE')
split_date = train_df['DATE'].quantile(0.85)  # 85% train, 15% val

train = train_df[train_df['DATE'] <= split_date]
val   = train_df[train_df['DATE'] > split_date]

print(f"\nРазделение: Train {train.shape}, Val {val.shape}")

# ========== ВАШИ ПРИЗНАКИ (без изменений) ==========
features = [
    'log_cpm',
    'month_sin', 'month_cos',
    'dow_sin', 'dow_cos'
]

cat_features = ['CHANNEL_NAME']

# ========== ОПТИМИЗИРОВАННАЯ МОДЕЛЬ ==========
model = CatBoostRegressor(
    iterations=1500,      # чуть меньше
    depth=7,             # чуть мельче
    learning_rate=0.03,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=200,
    early_stopping_rounds=150,
    use_best_model=True
)

model.fit(
    train[features + cat_features],
    train['log_views'],
    eval_set=(val[features + cat_features], val['log_views']),
    cat_features=cat_features,
    verbose=200
)

# ========== ВАЛИДАЦИЯ ==========
val_pred_log = model.predict(val[features + cat_features])
val_pred = np.expm1(val_pred_log)
val_true = np.expm1(val['log_views'])

rmsle = np.sqrt(
    np.mean((np.log1p(val_pred) - np.log1p(val_true)) ** 2)
)

print(f'\nRMSLE на валидации: {rmsle:.4f}')

# ========== АНАЛИЗ ОШИБОК ДЛЯ ПОСТОБРАБОТКИ ==========
print("\n" + "="*60)
print("АНАЛИЗ ОШИБОК ДЛЯ ПОСТОБРАБОТКИ:")
print("="*60)

# Группируем по CPM для анализа
val_analysis = pd.DataFrame({
    'CPM': val['CPM'].values,
    'true': val_true,
    'pred': val_pred,
    'error': val_pred - val_true  # положительное = завысили
})

# Делаем 8 групп по CPM с обработкой ошибок
try:
    val_analysis['cpm_group'] = pd.qcut(val_analysis['CPM'], q=min(8, len(val_analysis)//2), duplicates='drop')
except Exception as e:
    print(f"Ошибка при qcut: {e}, используем обычный cut")
    val_analysis['cpm_group'] = pd.cut(val_analysis['CPM'], bins=5)

print(f"{'CPM группа':<20} {'Средняя ошибка':<15} {'Истина':<10} {'Предсказание':<12}")
print("-" * 60)

group_stats = []
for group in sorted(val_analysis['cpm_group'].unique()):
    mask = val_analysis['cpm_group'] == group
    group_data = val_analysis[mask]
    
    avg_cpm = float(group_data['CPM'].mean())
    avg_true = float(group_data['true'].mean())
    avg_pred = float(group_data['pred'].mean())
    avg_error = float(group_data['error'].mean())
    error_pct = float((avg_error / avg_true * 100) if avg_true > 0 else 0)
    
    # Безопасное получение границ группы
    cpm_min = float(group.left) if group.left is not None else 0.0
    cpm_max = float(group.right) if group.right is not None else float('inf')
    
    group_stats.append({
        'cpm_min': cpm_min,
        'cpm_max': cpm_max,
        'avg_cpm': avg_cpm,
        'avg_true': avg_true,
        'avg_pred': avg_pred,
        'avg_error': avg_error,
        'error_pct': error_pct
    })
    
    # Форматируем для красивого вывода
    left_str = f"{group.left:.1f}" if group.left is not None else "0.0"
    right_str = f"{group.right:.1f}" if group.right is not None else "∞"
    print(f"{f'{left_str}-{right_str}':<20} {avg_error:<15.0f} ({error_pct:+.1f}%) {avg_true:<10.0f} {avg_pred:<12.0f}")

# ========== ИНТЕЛЛЕКТУАЛЬНАЯ ПОСТОБРАБОТКА ==========
def smart_postprocessing(predictions, cpms, group_stats):
    """
    Умная постобработка на основе анализа ошибок в валидации
    """
    corrected = predictions.copy()
    
    # Применяем коррекции на основе анализа групп CPM
    for i in range(len(corrected)):
        cpm = cpms.iloc[i]
        
        # Находим к какой группе относится этот CPM
        for stats in group_stats:
            # Проверяем попадание в диапазон (учитываем inf для последней группы)
            if stats['cpm_max'] == float('inf'):
                if cpm >= stats['cpm_min']:
                    # Корректируем на основе средней ошибки для этой группы
                    error_pct = stats['error_pct']
                    
                    if error_pct > 5:  # завышали более чем на 5%
                        correction_factor = 1.0 - (error_pct / 100)
                        corrected[i] *= max(correction_factor, 0.7)
                    elif error_pct < -5:  # занижали более чем на 5%
                        correction_factor = 1.0 - (error_pct / 100)
                        corrected[i] *= min(correction_factor, 1.3)
                    break
            else:
                if stats['cpm_min'] <= cpm < stats['cpm_max']:
                    # Корректируем на основе средней ошибки для этой группы
                    error_pct = stats['error_pct']
                    
                    if error_pct > 5:  # завышали более чем на 5%
                        correction_factor = 1.0 - (error_pct / 100)
                        corrected[i] *= max(correction_factor, 0.7)
                    elif error_pct < -5:  # занижали более чем на 5%
                        correction_factor = 1.0 - (error_pct / 100)
                        corrected[i] *= min(correction_factor, 1.3)
                    break
    
    # Дополнительная физическая коррекция
    for i in range(len(corrected)):
        cpm = cpms.iloc[i]
        
        # Физические ограничения
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

# ========== ПРЕДСКАЗАНИЕ НА ТЕСТЕ ==========
print("\n" + "="*60)
print("ПРЕДСКАЗАНИЕ НА ТЕСТЕ:")
print("="*60)

test_pred_log = model.predict(test_df[features + cat_features])
test_pred_raw = np.expm1(test_pred_log)

# Применяем умную постобработку
test_final = smart_postprocessing(test_pred_raw, test_df['CPM'], group_stats)

# ========== СМЕШИВАНИЕ С ФИЗИЧЕСКОЙ МОДЕЛЬЮ ==========
if rmsle > 0.90:
    print("\nДобавляем физическую модель для стабилизации...")
    
    physics_pred = 1000 / (test_df['CPM'] ** 0.65 + 5) + 20
    test_final = (test_final * 0.7 + physics_pred * 0.3).astype(int)
    
    print("Применено смешивание 70% ML + 30% физика")

test_df['VIEWS'] = test_final

# ========== ПРОВЕРКА РЕЗУЛЬТАТОВ ==========
print("\n" + "="*60)
print("ПРОВЕРКА РЕЗУЛЬТАТОВ:")
print("="*60)

# 1. Проверяем обратную зависимость
print("\n1. Проверка обратной зависимости:")

# Группируем тестовые предсказания по CPM
try:
    test_df['cpm_group'] = pd.qcut(test_df['CPM'], q=min(8, len(test_df)//2), duplicates='drop')
except:
    test_df['cpm_group'] = pd.cut(test_df['CPM'], bins=5)

prev_views = None
for group in sorted(test_df['cpm_group'].unique()):
    mask = test_df['cpm_group'] == group
    if mask.any():
        avg_cpm = test_df.loc[mask, 'CPM'].mean()
        avg_views = test_df.loc[mask, 'VIEWS'].mean()
        count = mask.sum()
        
        trend = ""
        if prev_views is not None:
            if avg_views < prev_views:
                trend = "↓ (правильно)"
            else:
                trend = f"↑ (нарушение: {avg_views-prev_views:.0f})"
        
        # Форматируем вывод
        left_str = f"{group.left:.1f}" if group.left is not None else "0.0"
        right_str = f"{group.right:.1f}" if group.right is not None else "∞"
        
        print(f"  CPM {left_str}-{right_str}: avg CPM={avg_cpm:5.1f}, VIEWS={avg_views:5.0f} {trend}")
        prev_views = avg_views

# 2. Статистика
print(f"\n2. Статистика предсказаний:")
print(test_df['VIEWS'].describe())

# 3. Сравнение распределений
train_median = float(train_df['VIEWS'].median())
test_median = float(test_df['VIEWS'].median())
median_diff_pct = float(abs(test_median - train_median) / train_median * 100 if train_median > 0 else 0)

print(f"\n3. Сравнение с тренировочными данными:")
print(f"   Train VIEWS медиана: {train_median:.0f}")
print(f"   Test VIEWS медиана:  {test_median:.0f}")
print(f"   Разница: {median_diff_pct:.1f}%")

if median_diff_pct > 20:
    print(f"   ⚠️  Большая разница! Калибруем...")
    # Калибруем медиану
    calibration_factor = train_median / test_median if test_median > 0 else 1.0
    if 0.7 < calibration_factor < 1.3:
        test_df['VIEWS'] = (test_df['VIEWS'] * calibration_factor).astype(int)
        print(f"   Применена калибровка: factor = {calibration_factor:.3f}")

# ========== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==========
result = test_df[['CPM', 'CHANNEL_NAME', 'DATE', 'VIEWS']]
output_file = 'optimized_final_predictions.csv'
result.to_csv(output_file, index=False)

print(f'\n✅ Результаты сохранены в {output_file}')

# ========== СОХРАНЕНИЕ МОДЕЛИ ==========
print("\n" + "="*60)
print("СОХРАНЕНИЕ МОДЕЛИ:")
print("="*60)

# 1. Сохраняем CatBoost модель
model.save_model('models/catboost_model.cbm')
print('✅ Модель CatBoost сохранена: models/catboost_model.cbm')

# 2. Сохраняем group_stats (уже все значения приведены к float)
serializable_stats = []
for stat in group_stats:
    serializable_stats.append({
        'cpm_min': stat['cpm_min'],
        'cpm_max': stat['cpm_max'],
        'avg_cpm': stat['avg_cpm'],
        'avg_true': stat['avg_true'],
        'avg_pred': stat['avg_pred'],
        'avg_error': stat['avg_error'],
        'error_pct': stat['error_pct']
    })

with open('models/group_stats.json', 'w', encoding='utf-8') as f:
    json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
print('✅ Статистики групп сохранены: models/group_stats.json')

# 3. Сохраняем model_info
model_info = {
    'features': features,
    'cat_features': cat_features,
    'train_median': train_median,
    'test_median': test_median,
    'rmsle': float(rmsle)
}

with open('models/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print('✅ Информация о модели сохранена: models/model_info.pkl')

# 4. Сохраняем метрики в читаемом виде
metrics = {
    'validation_rmsle': float(rmsle),
    'train_samples': len(train),
    'validation_samples': len(val),
    'test_samples': len(test_df),
    'train_median_views': train_median,
    'test_median_views': test_median
}

with open('models/metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print('✅ Метрики сохранены: models/metrics.json')

print(f'\n✅ Все файлы успешно сохранены в папке models/')
print(f'   ├── catboost_model.cbm')
print(f'   ├── group_stats.json')
print(f'   ├── model_info.pkl')
print(f'   └── metrics.json')
