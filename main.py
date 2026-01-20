import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
train_df = pd.read_csv(
    'data/doc-1766653704.csv',
    usecols=[' CPM', ' VIEWS', ' CHANNEL_NAME', ' DATE']
)

test_df = pd.read_csv('data/doc-1768808876.csv')

# Чистим названия колонок
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

print(train_df.shape, test_df.shape)
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
train_df['log_cpm'] = np.log1p(train_df['CPM'])
test_df['log_cpm'] = np.log1p(test_df['CPM'])

train_df['log_views'] = np.log1p(train_df['VIEWS'])
train_df = train_df.sort_values('DATE')

split_date = train_df['DATE'].quantile(0.8)

train = train_df[train_df['DATE'] <= split_date]
val   = train_df[train_df['DATE'] > split_date]

print(train.shape, val.shape)
features = [
    'log_cpm',
    'month_sin', 'month_cos',
    'dow_sin', 'dow_cos'
]

cat_features = ['CHANNEL_NAME']
model = CatBoostRegressor(
    iterations=2000,
    depth=8,
    learning_rate=0.03,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=200,
    early_stopping_rounds=200
)

model.fit(
    train[features + cat_features],
    train['log_views'],
    eval_set=(val[features + cat_features], val['log_views']),
    cat_features=cat_features,
    use_best_model=True
)
val_pred_log = model.predict(val[features + cat_features])

val_true = np.expm1(val['log_views'])
val_pred = np.expm1(val_pred_log)

rmsle = np.sqrt(
    np.mean((np.log1p(val_pred) - np.log1p(val_true)) ** 2)
)

print(f'RMSLE: {rmsle:.4f}')
test_pred_log = model.predict(test_df[features + cat_features])
test_pred = np.expm1(test_pred_log)

test_df['VIEWS'] = (
    np.round(test_pred)
    .clip(min=0)
    .astype(int)
)
result = test_df[['CPM', 'CHANNEL_NAME', 'DATE', 'VIEWS']]
result.to_csv('catboost_predictions.csv', index=False)

print('Saved to catboost_predictions.csv')
print(result['VIEWS'].describe())
