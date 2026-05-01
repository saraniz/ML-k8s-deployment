import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/creditcard.csv')

#basic
print(df.columns)
print(df.head())
print(df.tail())
print(df.shape)
# print(df.info())
# print(df.describe())

#Target Variable Distribution
print(df['Class'].value_counts()) #take the count of distinct values in Class column
print(df['Class'].value_counts(normalize=True)*100)

sns.countplot(x='Class', data=df)
plt.title("Fraud vs Non-Fraud")
plt.show()

#handling missing values
print(df.isnull().sum().sum()) #take the sum of null values in each column(first sum), then get the sum of null values of all columns (second sum)
print(df.isnull().sum().sort_values(ascending=False))

#handling duplicat values
print("Duplicates:", df.duplicated().sum())

df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

#Feature Overview
print(df[['Time', 'Amount']].describe())

#amount distribution analysis
plt.figure(figsize=(8,5)) 
sns.histplot(df['Amount'], bins=50) # bins=50 → splits data into 50 intervals (bars)
plt.title("Transaction Amount Distribution")
plt.show()

df['Log_Amount'] = np.log1p(df['Amount'])

sns.histplot(df['Log_Amount'], bins=50)
plt.title("Log Amount Distribution")
plt.show()

#Fraud vs Non-Fraud Amount Comparison
sns.boxplot(x='Class', y='Amount', data=df)
plt.title("Amount vs Fraud")
plt.show()

sns.boxplot(x='Class', y='Log_Amount', data=df)
plt.title("Log Amount vs Fraud")
plt.show()

#time analysis
plt.figure(figsize=(10,5))
sns.histplot(df['Time'], bins=50)
plt.title("Transaction Time Distribution")
plt.show()

fraud = df[df['Class'] == 1]

plt.figure(figsize=(10,5))
sns.histplot(fraud['Time'], bins=50, color='red')
plt.title("Fraud Over Time")
plt.show()

plt.figure(figsize=(12,8))
corr = df.corr()

sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

features = ['V10', 'V12', 'V14', 'V17']

for col in features:
    plt.figure(figsize=(8,4))
    sns.kdeplot(df[df['Class']==0][col], label='Normal')
    sns.kdeplot(df[df['Class']==1][col], label='Fraud')
    plt.title(f"{col} Distribution")
    plt.legend()
    plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(df['V10'])
plt.title("Outlier Check V10")
plt.show()

fraud_counts = df['Class'].value_counts()

plt.pie(fraud_counts, labels=['Non-Fraud', 'Fraud'], autopct='%1.2f%%')
plt.title("Class Imbalance")
plt.show()

df['Amount_per_Time'] = df['Amount'] / (df['Time'] + 1)
df['Log_Amount'] = np.log1p(df['Amount'])

print(df.info())
print(df.describe())
print(df['Class'].value_counts())