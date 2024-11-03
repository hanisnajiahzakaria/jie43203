import pandas as pd
import streamlit as st

data = pd.read_csv('https://raw.githubusercontent.com/hanisnajiahzakaria/jie43203/refs/heads/main/housing.csv')
data.head()
print(data.shape)

st.write(data)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,  r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

a = data.isnull().any()
b = data.isnull().sum()

print(a)
print()
print(b)

# handle missing data (delete column in df)
df = pd.DataFrame(data)
del df['total_bedrooms']


# check again the null value to ensure it already drop the column
df.isnull().any()

# change the category into number categorical
df['ocean_proximity'] = df['ocean_proximity'].replace({'NEAR BAY': 1, '<1H OCEAN': 2,
                                                       'INLAND': 2, 'NEAR OCEAN': 1,'ISLAND':1})


df.info()

#Creating subplot of each column with its own scale
red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')

fig, axs = plt.subplots(1, len(df.columns), figsize=(20,10))

# detecting outliers in every column in dataset
for i, ax in enumerate(axs.flat):
    ax.boxplot(df.iloc[:,i], flierprops=red_circle)
    ax.set_title(df.columns[i], fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)

plt.tight_layout()
