import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import os
def download():
    import kagglehub
    kagglehub.login()
    rainfall_path = kagglehub.dataset_download("act18l/hongkongrainfall")
    air_quality_path = kagglehub.dataset_download("samsonlo/hong-kong-air-quality-data-201019")

rainfall_path = r"C:/Users/waila/.cache/kagglehub/datasets/act18l/hongkongrainfall/versions/2/hongkong.csv"
air_quality_path = r"C:/Users/waila/.cache/kagglehub/datasets/samsonlo/hong-kong-air-quality-data-201019/versions/1"


files = []
for file in os.listdir(air_quality_path):
    df = pd.read_excel(os.path.join(air_quality_path, file), header=10)
    df.columns = df.iloc[0]
    df = df[1:]
    files.append(df)

air_quality_df = pd.concat(files, ignore_index=True)
rainfall_df= pd.read_csv(rainfall_path,encoding='cp874')

air_quality_df.replace('N.A.', np.nan, inplace=True)
air_quality_df['DATE'] = pd.to_datetime(air_quality_df['DATE'], errors='coerce')
air_quality_df['HOUR'] = pd.to_numeric(air_quality_df['HOUR'])
air_quality_df['HOUR'] = air_quality_df['HOUR'].apply(lambda x: x - 1 if x == 24 else x - 1)

rainfall_df['DATE'] = pd.to_datetime(rainfall_df[['year', 'month', 'day']])
rainfall_df['rainfall'] = pd.to_numeric(rainfall_df['rainfall'], errors='coerce')

final_df = pd.merge(air_quality_df, rainfall_df, on='DATE', how='left')

final_df['day_of_week'] = final_df['DATE'].dt.dayofweek

feature_cols = ['temparature', 'windspeed', 'rainfall', 'pressure', 'humidity', 'HOUR', 'month', 'day_of_week']
target_cols = ['NO2', 'RSP', 'O3']
final_df = final_df.sort_values(['STATION', 'DATE'])
final_df['NO2_lag24'] = final_df.groupby('STATION')['NO2'].shift(24)
final_df['RSP_lag24'] = final_df.groupby('STATION')['RSP'].shift(24)
final_df['O3_lag24'] = final_df.groupby('STATION')['O3'].shift(24)
feature_cols = [
    'temparature', 'windspeed', 'rainfall', 'pressure', 'humidity', 
    'HOUR', 'month', 'day_of_week', 
    'NO2_lag24', 'RSP_lag24', 'O3_lag24'
]

df_model = final_df.dropna(subset=target_cols + ['NO2_lag24', 'RSP_lag24', 'O3_lag24']).copy()

df_model[feature_cols] = df_model[feature_cols].interpolate(method='linear').fillna(0)

X = df_model[feature_cols]
y = df_model[target_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [
            'temparature', 'windspeed', 'rainfall', 'pressure', 'humidity', 
            'NO2_lag24', 'RSP_lag24', 'O3_lag24'
        ]),
        ('cat', 'passthrough', ['HOUR', 'month', 'day_of_week'])
    ])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(XGBRegressor(n_estimators=100)))
])


model_pipeline.fit(X_train, y_train)


from sklearn.metrics import r2_score
y_pred = model_pipeline.predict(X_test)
for i, col in enumerate(target_cols):
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"{col} score: {r2:.4f}")

importance = model_pipeline.named_steps['regressor'].estimators_[0].feature_importances_
for name, imp in zip(feature_cols, importance):
    print(f"{name}: {imp:.4f}")


import matplotlib.pyplot as plt
import seaborn as sns


fig = plt.figure(figsize=(10, 15), constrained_layout=True)


ax1 = plt.subplot(311)
ax1.plot(y_test.iloc[:100, 0].values, label='Actual NO2', marker='o', linestyle='--', alpha=0.7)
ax1.plot(y_pred[:100, 0], label='Predicted NO2', marker='x', alpha=0.7)
ax1.set_title('NO2 Prediction vs Actual (First 100 Samples)', fontsize=14, pad=10)
ax1.legend()


ax2 = plt.subplot(312)
feat_importances = pd.Series(importance, index=feature_cols)
feat_importances.nlargest(11).plot(kind='barh', color='skyblue', ax=ax2)
ax2.set_title('Feature Importance for NO2 Prediction', fontsize=14, pad=10)
ax2.invert_yaxis()
plt.subplots_adjust(left=0.2) 


ax3 = plt.subplot(313)
residuals = y_test.iloc[:, 0] - y_pred[:, 0]
sns.histplot(residuals, kde=True, color='purple', ax=ax3)
ax3.set_title('Residuals Distribution for NO2', fontsize=14, pad=10)
ax3.set_xlabel('Error (Actual - Predicted)')

plt.savefig("result.png", dpi=300, bbox_inches='tight')
plt.show()