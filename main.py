from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


# 한글 폰트 설정하기
fe = fm.FontEntry(fname='MaruBuri-Regular.otf', name='MaruBuri')
fm.fontManager.ttflist.insert(0, fe)
plt.rc('font', family='MaruBuri')


# train, test 데이터를 불러옵니다.
train = pd.read_csv('C:/Users/MSI/Desktop/Pytorch/open/train.csv')
test = pd.read_csv('C:/Users/MSI/Desktop/Pytorch/open/test.csv')

train


def plot_histogram(data, columns_to_plot, cols=2, figsize=(10, 5)):
    # 해당 열이 데이터 내 존재하는 지 확인합니다.
    valid_columns = [col for col in columns_to_plot if col in data.columns]
    if not valid_columns:
        raise ValueError("해당 열이 존재하지 않습니다.")

    # 서브플롯을 생성하기 위해 행과 열의 수를 확인합니다.
    num_vars = len(valid_columns)
    rows = math.ceil(num_vars / cols)

    # 서브플롯을 생성합니다.
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(valid_columns):
        sns.histplot(x=data[col], ax=axes[i])
        axes[i].set_title(col)

    # 빈 서브플롯을 삭제합니다.
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# 그래프를 생성할 열을 담은 리스트 입니다.
columns_to_plot = ['배터리용량']

# 선언한 함수를 사용하여 그래프를 생성합니다.
plot_histogram(data=train, columns_to_plot=columns_to_plot,
               cols=2, figsize=(10, 5))


def plot_categorical(data, columns_to_plot, cols=3, figsize=(10, 5)):
    # 해당열이 데이터 내 존재하는지 확인합니다.
    valid_columns = [col for col in columns_to_plot if col in data.columns]
    if not valid_columns:
        raise ValueError("해당 열이 존재하지 않습니다.")

    # 서브플롯을 생성하기 위해 행과 열의 수를 확인합니다.
    num_vars = len(valid_columns)
    rows = math.ceil(num_vars / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(valid_columns):
        sns.countplot(x=data[col], ax=axes[i], palette="viridis",
                      order=data[col].value_counts().index)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].tick_params(axis='x', rotation=45)

    # 빈 서비플롯을 삭제합니다.
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # 그래프를 생성할 열을 담은 리스트 입니다.
columns_to_plot = ['차량상태']

# 선언한 함수를 사용하여 그래프를 생성합니다.
plot_categorical(data=train, columns_to_plot=columns_to_plot)

# 제조사의 모델별 가격의 평균으로 그룹화 합니다.
grouped_data = train.groupby(['제조사', '모델'])['가격(백만원)'].mean().reset_index()

# 더 나은 그래프를 그리기 위해 제조사별로 정렬합니다.
grouped_data = grouped_data.sort_values(
    by=['제조사', '가격(백만원)'], ascending=[True, False])

# 그래프 전체의 크기를 나타냅니다.
plt.figure(figsize=(15, 8))

# 유니크한 제조사 명을 정의합니다.
brands = grouped_data['제조사'].unique()
colors = plt.cm.tab20.colors[:len(brands)]

# 제조사별로 모델의 가격을 바 그래프 형태로 나타냅니다.
for i, brand in enumerate(brands):
    brand_data = grouped_data[grouped_data['제조사'] == brand]
    plt.bar(
        brand_data['모델'],
        brand_data['가격(백만원)'],
        label=brand,
        color=colors[i % len(colors)]
    )

# 그래프의 형태를 정의합니다.
plt.title('제조사 별 모델 가격 분포', fontsize=16)
plt.xlabel('모델', fontsize=12)
plt.ylabel('가격(백만원)', fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.legend(title='제조사', fontsize=10, loc='upper right')
plt.tight_layout()
plt.show()

# 데이터 내 결측치를 확인합니다.
train.isna().sum()

# 데이터 내 결측치를 train데이터의 평균값을 사용하여 대체합니다.
train['배터리용량'].fillna(train['배터리용량'].mean(), inplace=True)
test['배터리용량'].fillna(train['배터리용량'].mean(), inplace=True)

# 학습과 예측을 위해 데이터를 분리합니다.
x_train = train.drop(['ID', '가격(백만원)'], axis=1)
y_train = train['가격(백만원)']

x_test = test.drop('ID', axis=1)

# 범주형 변수에 대해 레이블 인코딩을 적용합니다.
categorical_features = [
    col for col in x_train.columns if x_train[col].dtype == 'object']

for i in categorical_features:
    le = LabelEncoder()
    le = le.fit(x_train[i])
    x_train[i] = le.transform(x_train[i])

    for case in np.unique(x_test[i]):
        if case not in le.classes_:
            le.classes_ = np.append(le.classes_, case)
    x_test[i] = le.transform(x_test[i])

print(x_train.head(3))

# 모델을 선언하고, 학습을 진행합니다.
model = DecisionTreeRegressor()
model.fit(x_train, y_train)

# 예측을 수행합니다.
pred = model.predict(x_test)

# sample_submission 파일을 불러옵니다.
submit = pd.read_csv('C:/Users/MSI/Desktop/Pytorch/open/sample_submission.csv')

# 예측한 결과를 submit에 저장합니다.
submit['가격(백만원)'] = pred
submit.head()

# 제출을 위해 baseline_submission.csv란 이름으로 CSV 파일을 저장합니다.
submit.to_csv('./baseline_submission.csv', index=False)
