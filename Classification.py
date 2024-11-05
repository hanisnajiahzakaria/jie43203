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
st.pyplot(plt.gcf())

# find iqr
households_percentile25 = df['households'].quantile(0.25)
households_percentile75 = df['households'].quantile(0.75)
iqr = households_percentile75 - households_percentile25

# find upper and lower limit
households_upper_limit = households_percentile75 + 1.5 * iqr
households_lower_limit = households_percentile25 - 1.5 * iqr

# finding outliers for households
df[df['households'] > households_upper_limit]
df[df['households'] < households_lower_limit]

new_df = df[df['households'] < households_upper_limit]

# find iqr
income_percentile25 = new_df['median_income'].quantile(0.25)
income_percentile75 = new_df['median_income'].quantile(0.75)
iqr = income_percentile75 - income_percentile25

# find upper and lower limit
income_upper_limit = income_percentile75 + 1.5 * iqr
income_lower_limit = income_percentile25 - 1.5 * iqr

# finding outliers for median_income
new_df[new_df['median_income'] > income_upper_limit]
new_df[new_df['median_income'] < income_lower_limit]

new_df = new_df[new_df['median_income'] < income_upper_limit]

# find iqr
totroom_percentile25 = new_df['total_rooms'].quantile(0.25)
totroom_percentile75 = new_df['total_rooms'].quantile(0.75)
iqr = totroom_percentile75 - totroom_percentile25

# find upper and lower limit
totroom_upper_limit = totroom_percentile75 + 1.5 * iqr
totroom_lower_limit = totroom_percentile25 - 1.5 * iqr

# finding outliers for total_rooms
new_df[new_df['total_rooms'] > totroom_upper_limit]
new_df[new_df['total_rooms'] < totroom_lower_limit]

new_df = new_df[new_df['total_rooms'] < totroom_upper_limit]

# find iqr
populations_percentile25 = new_df['population'].quantile(0.25)
populations_percentile75 = new_df['population'].quantile(0.75)
iqr = populations_percentile75 - populations_percentile25

# find upper and lower limit
populations_upper_limit = populations_percentile75 + 1.5 * iqr
populations_lower_limit = populations_percentile25 - 1.5 * iqr

# finding outliers for population
new_df[new_df['population'] > populations_upper_limit]
new_df[new_df['population'] < populations_lower_limit]

new_df = new_df[new_df['population'] < populations_upper_limit]

# check wether the outliers already been removed

#Creating subplot of each column with its own scale
red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')

fig, axs = plt.subplots(1, len(new_df.columns), figsize=(20,10))

# detecting outliers in every column in dataset
for i, ax in enumerate(axs.flat):
    ax.boxplot(new_df.iloc[:,i], flierprops=red_circle)
    ax.set_title(new_df.columns[i], fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)

plt.tight_layout()

# checking duplication data in dataset
new_df.duplicated().sum()

print(new_df.shape)

new_df.describe()


