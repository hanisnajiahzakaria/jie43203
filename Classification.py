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

# Page title and divider
st.title("California Housing Dataset Analysis 📊")
st.divider()

# Data overview with metrics
st.subheader("Data Overview 📈")
col1, col2, col3 = st.columns(3)
col1.metric("Total Entries", data.shape[0])
col2.metric("Total Features", data.shape[1])
col3.metric("Ocean Proximity Categories", data['ocean_proximity'].nunique())

# Display dataset shape and sample data
st.subheader("Initial Dataset Overview")
st.write("Dataset shape:", data.shape)
st.write(data.head())

# Check and visualize missing values
st.subheader("Missing Values in Each Column")
missing_info = pd.DataFrame({"Missing": data.isnull().any(), "Count": data.isnull().sum()})
st.write(missing_info)

# Visualize missing values as a bar chart
st.subheader("Missing Values Count by Column")
missing_counts = data.isnull().sum()
plt.figure(figsize=(10, 6))
missing_counts[missing_counts > 0].plot(kind='bar', color='salmon')
plt.title("Missing Values Count per Column")
plt.xlabel("Columns")
plt.ylabel("Number of Missing Values")
st.pyplot(plt.gcf())

# Handling missing data by dropping 'total_bedrooms' column
df = data.drop(columns=['total_bedrooms'])

# Display data info after handling missing values
st.subheader("Data Information After Dropping 'total_bedrooms'")
buffer = st.empty()  # To handle print output in Streamlit
buffer.text(df.info())

# Encode 'ocean_proximity' to numeric values
df['ocean_proximity'] = df['ocean_proximity'].replace({
    'NEAR BAY': 1, '<1H OCEAN': 2, 'INLAND': 2, 'NEAR OCEAN': 1, 'ISLAND': 1
})

# Sidebar filters for data exploration
st.sidebar.subheader("Filter Data")
age_filter = st.sidebar.slider("Select Housing Median Age", int(df['housing_median_age'].min()), int(df['housing_median_age'].max()))
income_filter = st.sidebar.slider("Select Median Income", float(df['median_income'].min()), float(df['median_income'].max()))

filtered_df = df[(df['housing_median_age'] <= age_filter) & (df['median_income'] <= income_filter)]
st.subheader("Filtered Data")
st.write(filtered_df)

# Boxplots for outlier detection
st.subheader("Boxplots for Outlier Detection")
red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')
fig, axs = plt.subplots(1, len(df.columns), figsize=(20, 10))
for i, ax in enumerate(axs.flat):
    ax.boxplot(df.iloc[:, i], flierprops=red_circle)
    ax.set_title(df.columns[i], fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()
st.pyplot(fig)

# Outlier Removal
st.subheader("Outliers Removal Process")

# Apply IQR method to remove outliers for households, median_income, total_rooms, and population
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    return data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]

new_df = df.copy()
for col in ['households', 'median_income', 'total_rooms', 'population']:
    new_df = remove_outliers(new_df, col)

# Check outliers removal with boxplots
st.subheader("Outliers Removal: Boxplots After Filtering")
fig, axs = plt.subplots(1, len(new_df.columns), figsize=(20, 10))
for i, ax in enumerate(axs.flat):
    ax.boxplot(new_df.iloc[:, i], flierprops=red_circle)
    ax.set_title(new_df.columns[i], fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()
st.pyplot(fig)

# Duplicate data check
st.subheader("Duplicate Data Check")
st.write("Number of duplicate rows in the dataset:", new_df.duplicated().sum())

# Final dataset shape and summary statistics
st.subheader("Final Dataset Shape and Summary Statistics")
st.write("Final dataset shape after outlier removal:", new_df.shape)
st.write(new_df.describe())

# Correlation heatmap
st.subheader("Feature Correlation Heatmap")
plt.figure(figsize=(10, 8))
sns.heatmap(new_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
st.pyplot(plt.gcf())

# Interactive histogram
st.subheader("Interactive Histogram")
column_to_plot = st.selectbox("Select a column to display its histogram:", new_df.columns)
plt.figure(figsize=(8, 5))
sns.histplot(new_df[column_to_plot], kde=True)
st.pyplot(plt.gcf())

# Pairplot for feature correlations
st.subheader("Pairplot for Feature Correlations")
pairplot_fig = sns.pairplot(new_df[['median_house_value', 'median_income', 'housing_median_age', 'total_rooms']], corner=True)
st.pyplot(pairplot_fig)

# Prediction model (example)
st.sidebar.subheader("Predict House Value")
# Example inputs for prediction (replace with actual model if trained)
median_income_input = st.sidebar.slider("Median Income", float(new_df['median_income'].min()), float(new_df['median_income'].max()))
housing_age_input = st.sidebar.slider("Housing Median Age", int(new_df['housing_median_age'].min()), int(new_df['housing_median_age'].max()))

# Placeholder for prediction (replace this with trained model prediction code)
# For instance, if model is `trained_model`, then use `prediction = trained_model.predict([[median_income_input, housing_age_input]])`
prediction = median_income_input * 1000  # This is a placeholder
st.sidebar.write(f"Predicted House Price: ${prediction:,.2f}")
