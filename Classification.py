import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load and display initial data
data = pd.read_csv('https://raw.githubusercontent.com/hanisnajiahzakaria/jie43203/refs/heads/main/housing.csv')

# Display dataset shape and sample data
st.title("California Housing Dataset Analysis", divider="red")
st.subheader("Initial Dataset Overview")
st.write("Dataset shape:", data.shape)
st.write(data.head())

# Check for missing values
a = data.isnull().any()
b = data.isnull().sum()
st.subheader("Missing Values in Each Column")
st.write("Columns with missing values:", a)
st.write("Count of missing values:", b)

# Handling missing data by dropping 'total_bedrooms' column
df = pd.DataFrame(data)
df.drop(columns=['total_bedrooms'], inplace=True)

# Display column types and summary
st.subheader("Data Information After Dropping 'total_bedrooms'")
st.write(df.info())

# Encode 'ocean_proximity' to numeric values
df['ocean_proximity'] = df['ocean_proximity'].replace({
    'NEAR BAY': 1, '<1H OCEAN': 2, 'INLAND': 2, 'NEAR OCEAN': 1, 'ISLAND': 1
})

# Plot boxplots for each column to identify outliers
st.subheader("Boxplots for Outlier Detection")
st.write("Identifying outliers in each feature:")

red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')
fig, axs = plt.subplots(1, len(df.columns), figsize=(20, 10))
for i, ax in enumerate(axs.flat):
    ax.boxplot(df.iloc[:, i], flierprops=red_circle)
    ax.set_title(df.columns[i], fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()
st.pyplot(fig)

# Outlier Removal Process
# Calculate IQR and set thresholds for each column and filter data
# Households Outliers
households_percentile25 = df['households'].quantile(0.25)
households_percentile75 = df['households'].quantile(0.75)
iqr = households_percentile75 - households_percentile25
households_upper_limit = households_percentile75 + 1.5 * iqr

# Filter data based on households
new_df = df[df['households'] < households_upper_limit]

# Median Income Outliers
income_percentile25 = new_df['median_income'].quantile(0.25)
income_percentile75 = new_df['median_income'].quantile(0.75)
iqr = income_percentile75 - income_percentile25
income_upper_limit = income_percentile75 + 1.5 * iqr

# Filter data based on median_income
new_df = new_df[new_df['median_income'] < income_upper_limit]

# Total Rooms Outliers
totroom_percentile25 = new_df['total_rooms'].quantile(0.25)
totroom_percentile75 = new_df['total_rooms'].quantile(0.75)
iqr = totroom_percentile75 - totroom_percentile25
totroom_upper_limit = totroom_percentile75 + 1.5 * iqr

# Filter data based on total_rooms
new_df = new_df[new_df['total_rooms'] < totroom_upper_limit]

# Population Outliers
populations_percentile25 = new_df['population'].quantile(0.25)
populations_percentile75 = new_df['population'].quantile(0.75)
iqr = populations_percentile75 - populations_percentile25
populations_upper_limit = populations_percentile75 + 1.5 * iqr

# Filter data based on population
new_df = new_df[new_df['population'] < populations_upper_limit]

# Check outliers removal
st.subheader("Outliers Removal: Boxplots After Filtering")
st.write("After removing outliers, the boxplots show less extreme values:")

fig, axs = plt.subplots(1, len(new_df.columns), figsize=(20, 10))
for i, ax in enumerate(axs.flat):
    ax.boxplot(new_df.iloc[:, i], flierprops=red_circle)
    ax.set_title(new_df.columns[i], fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()
st.pyplot(fig)

# Check for duplicate rows
st.subheader("Duplicate Data Check")
st.write("Number of duplicate rows in the dataset:", new_df.duplicated().sum())

# Final dataset shape and summary statistics
st.subheader("Final Dataset Shape and Summary Statistics")
st.write("Final dataset shape after outlier removal:", new_df.shape)
st.write("Statistical summary of the cleaned dataset:")
st.write(new_df.describe())

