# 1. 라이브러리 임포트
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# 2. 데이터 로드
train = pd.read_csv("/train.csv")
test = pd.read_csv("/test.csv")
submission = pd.read_csv("/sample_submission.csv")

# 3. 데이터 전처리
X = train.drop(['가격(백만원)', 'ID'], axis=1)
y = train['가격(백만원)']

# 결측치 처리
X['배터리용량'] = X['배터리용량'].fillna(X['배터리용량'].median())
test['배터리용량'] = test['배터리용량'].fillna(test['배터리용량'].median())

# 파생 변수 추가
X['에너지효율'] = X['배터리용량'] / (X['주행거리(km)'] + 1e-5)
X['차량나이'] = 2025 - X['연식(년)']
test['에너지효율'] = test['배터리용량'] / (test['주행거리(km)'] + 1e-5)
test['차량나이'] = 2025 - test['연식(년)']

# 원-핫 인코딩
X = pd.get_dummies(X, columns=['제조사', '모델', '차량상태', '구동방식', '사고이력'])
X_test = pd.get_dummies(test.drop(['ID'], axis=1), columns=[
                        '제조사', '모델', '차량상태', '구동방식', '사고이력'])
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# 데이터 분할
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 4. Optuna로 하이퍼파라미터 튜닝


def objective(trial):
    # 모델 선택
    model_name = trial.suggest_categorical('model', [
                                           'XGBoost', 'CatBoost', 'LightGBM', 'Random Forest', 'Extra Trees', 'Gradient Boosting'])

    if model_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
        }
        model = XGBRegressor(**params)

    elif model_name == 'CatBoost':
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1500, step=100),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'depth': trial.suggest_int('depth', 3, 12),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1, 10)
        }
        model = CatBoostRegressor(**params, loss_function='RMSE', verbose=0)

    elif model_name == 'LightGBM':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0)
        }
        model = LGBMRegressor(**params)

    elif model_name == 'Random Forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        model = RandomForestRegressor(**params)

    elif model_name == 'Extra Trees':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        model = ExtraTreesRegressor(**params)

    elif model_name == 'Gradient Boosting':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0)
        }
        model = GradientBoostingRegressor(**params)

    # 모델 학습
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    return rmse


# Optuna 최적화 실행
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 최적의 하이퍼파라미터 출력
print(f"Best trial: {study.best_trial.params}")

# 5. 최적 모델 학습 및 예측
# 최적 모델 학습 (Best 모델을 선택)
best_model_name = study.best_trial.params['model']
best_params = study.best_trial.params

if best_model_name == 'XGBoost':
    model = XGBRegressor(**best_params)
elif best_model_name == 'CatBoost':
    model = CatBoostRegressor(**best_params, loss_function='RMSE', verbose=0)
elif best_model_name == 'LightGBM':
    model = LGBMRegressor(**best_params)
elif best_model_name == 'Random Forest':
    model = RandomForestRegressor(**best_params)
elif best_model_name == 'Extra Trees':
    model = ExtraTreesRegressor(**best_params)
elif best_model_name == 'Gradient Boosting':
    model = GradientBoostingRegressor(**best_params)

# 최적 모델을 사용하여 학습
model.fit(X_train, y_train)

# 테스트 데이터 예측
final_test_pred = model.predict(X_test)

# 결과 값 클리핑 (예측 값이 가격 범위 내로 제한되도록)
lower_bound = y.min()
upper_bound = y.max()
final_test_pred = np.clip(final_test_pred, lower_bound, upper_bound)

# 제출 파일 생성
submission['가격(백만원)'] = final_test_pred
submission.to_csv("/predicted_submission_optuna.csv", index=False)
print("Predicted results saved to: predicted_submission_optuna.csv")
