import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

file_path = r'C:\Users\shino\Desktop\UczenieMaszynowe\UczenieMaszynowe_24\Lab 5\Salary_Data.csv'

data = pd.read_csv(file_path)

sns.set(style="whitegrid")
sns.histplot(data['Salary'], bins=10, kde=True)
plt.title('Rozkład wynagrodzeń')
plt.show()

sns.scatterplot(x='YearsExperience', y='Salary', data=data)
plt.title('Wynagrodzenie vs Doświadczenie')
plt.show()

X = data[['YearsExperience']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)



y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)



plt.scatter(X_train, y_train, color='blue', label='Rzeczywiste wartości (trening)')
plt.plot(X_train, y_train_pred, color='red', label='Przewidywane wartości (trening)')
plt.title('Wyniki zestawów treningowych')
plt.xlabel('Doświadczenie (lata)')
plt.ylabel('Wynagrodzenie')
plt.legend()
plt.show()

plt.scatter(X_test, y_test, color='blue', label='Rzeczywiste wartości (test)')
plt.plot(X_train, y_train_pred, color='red', label='Przewidywane wartości (trening)')
plt.title('Wyniki zestawów testowych')
plt.xlabel('Doświadczenie (lata)')
plt.ylabel('Wynagrodzenie')
plt.legend()
plt.show()
