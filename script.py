# %%
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# %%
housing = datasets.fetch_california_housing()

# %%
housing

# %%
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df.head()

# %%
dfTarget = pd.DataFrame(housing.target, columns=housing.target_names)
dfTarget.head()

# %%
x = df
y = dfTarget

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

# %%
x_train

# %%
y_train

# %%
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
x_train_scaled

# %%



reg = LinearRegression().fit(x_train_scaled, y_train)

# %%
result = reg.predict(x_test_scaled)
result

# %%
reg.score(x_test_scaled, y_test)


