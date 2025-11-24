print('Hello World')

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. 임의의 데이터 생성
np.random.seed(42)

# 독립변수 생성
n = 100
experience = np.random.uniform(0, 10, n)  # 경력 (0-10년)
education = np.random.choice([12, 14, 16, 18], n)  # 학력 (고졸, 전문대, 대졸, 석사)

# 종속변수 생성 (실제 관계식 + 노이즈)
# 연봉 = 3000 + 500*경력 + 200*학력 + 오차
salary = 3000 + 500 * experience + 200 * education + np.random.normal(0, 500, n)

# 2. 데이터프레임 생성
df = pd.DataFrame({
    'experience': experience,
    'education': education,
    'salary': salary
})

print("데이터프레임 미리보기:")
print(df.head())
print(f"\n데이터 shape: {df.shape}")
print("\n기술통계:")
print(df.describe())

# 3. 회귀분석 수행
X = df[['experience', 'education']]  # 독립변수
y = df['salary']  # 종속변수

model = LinearRegression()
model.fit(X, y)

# 4. 회귀분석 결과
print("\n=== 회귀분석 결과 ===")
print(f"절편 (Intercept): {model.intercept_:.2f}")
print(f"회귀계수 (Coefficients):")
print(f"  - experience: {model.coef_[0]:.2f}")
print(f"  - education: {model.coef_[1]:.2f}")
print(f"\nR² Score: {model.score(X, y):.4f}")

# 5. 예측
df['predicted_salary'] = model.predict(X)
df['residual'] = df['salary'] - df['predicted_salary']

print("\n예측값과 잔차:")
print(df[['salary', 'predicted_salary', 'residual']].head())

# 6. 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 실제값 vs 예측값
axes[0].scatter(df['salary'], df['predicted_salary'], alpha=0.5)
axes[0].plot([df['salary'].min(), df['salary'].max()], 
             [df['salary'].min(), df['salary'].max()], 
             'r--', lw=2)
axes[0].set_xlabel('Actual Salary')
axes[0].set_ylabel('Predicted Salary')
axes[0].set_title('Actual vs Predicted Salary')
axes[0].grid(True, alpha=0.3)

# 잔차 플롯
axes[1].scatter(df['predicted_salary'], df['residual'], alpha=0.5)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Salary')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. 새로운 데이터로 예측
new_data = pd.DataFrame({
    'experience': [5, 8],
    'education': [16, 18]
})
new_predictions = model.predict(new_data)

print("\n=== 새로운 데이터 예측 ===")
for i, pred in enumerate(new_predictions):
    print(f"경력 {new_data.iloc[i]['experience']}년, "
          f"학력 {new_data.iloc[i]['education']}년 → "
          f"예상 연봉: {pred:.2f}만원")
    
import statsmodels.api as sm

X_sm = sm.add_constant(X)
model_sm = sm.OLS(y, X_sm).fit()
print(model_sm.summary())

