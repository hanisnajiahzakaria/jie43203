import pandas as pd

data = pd.read_csv('https://raw.githubusercontent.com/hanisnajiahzakaria/jie43203/refs/heads/main/housing.csv')
data.head()
print(data.shape)

st.write(data)
