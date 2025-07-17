

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn as sk
import re
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data=pd.read_csv("/content/ApartmentRentPrediction.csv")

"""Check null values

"""

data.isnull().sum()

"""check distribution if normal or not"""

columns_to_clean = ['bathrooms','bedrooms','latitude','longitude']


#Remove null values from specific columns
data = data.dropna(subset=columns_to_clean)

plt.hist(data['bathrooms'], bins=10, color='blue', edgecolor='black')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of column_name')

data['bathrooms'] = np.where(data['bathrooms'] >=4, 3, data['bathrooms'])

plt.hist(data['bathrooms'], bins=10, color='blue', edgecolor='black')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of column_name')

sns.boxplot(x='bathrooms', data=data)
plt.xlabel('bathrooms')
plt.ylabel('Values')
plt.title('Boxplot of Column Name with Outliers Replaced by Max')
plt.show()

data['bathrooms'].fillna(data['bathrooms'].mean(), inplace=True)

sns.boxplot(x='bathrooms', data=data)
plt.xlabel('bathrooms')
plt.ylabel('Values')
plt.title('Boxplot of Bathrooms after null is Replaced by Max')
plt.show()

"""## Bedrooms"""

plt.hist(data['bedrooms'], bins=10, color='blue', edgecolor='black')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of bedrooms')
#normal

data['bedrooms'].fillna(data['bedrooms'].mean(), inplace=True)

plt.hist(data['bedrooms'], bins=10, color='blue', edgecolor='black')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of bedrooms after replace null with max value')

"""## Square_feet"""

plt.hist(data['square_feet'], bins=10, color='blue', edgecolor='black')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of square_feet')
# not normal

data['square_feet']=np.log(data['square_feet'])

plt.hist(data['square_feet'], bins=10, color='blue', edgecolor='black')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of square_feet')

sns.boxplot(x='square_feet', data=data)

# Adding labels and title
plt.xlabel('square_feet')
plt.ylabel('Values')
plt.title('Boxplot of sqaure_feet')

data['square_feet'] = np.where(data['square_feet'] >=7.6, 7.7, data['square_feet'])
data['square_feet'] = np.where(data['square_feet'] <=5.7, 5.7, data['square_feet'])

sns.boxplot(x='square_feet', data=data)
plt.xlabel('square_feet')
plt.ylabel('Values')
plt.title('Boxplot of square_feet with Outliers Replaced by Max')
plt.show()

plt.hist(data['square_feet'], bins=10, color='blue', edgecolor='black')


plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of square_feet')

"""## Longitude"""

plt.hist(data['longitude'], bins=10, color='blue', edgecolor='black')


plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of longitude')

data['longitude'] = np.where(data['longitude'] <=-122, -122, data['longitude'])

plt.hist(data['longitude'], bins=10, color='blue', edgecolor='black')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of longitude after replace null with max value')

"""## Latitude"""

plt.hist(data['latitude'], bins=10, color='blue', edgecolor='black')


plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of latitude')

data['latitude'] = np.where(data['latitude'] <=25, 25, data['latitude'])
data['latitude'] = np.where(data['latitude'] >=50, 50, data['latitude'])

sns.boxplot(x='latitude', data=data)
plt.xlabel('latitude')
plt.ylabel('Values')
plt.title('Boxplot of latitude after null is Replaced by Mean')
plt.show()

data.isnull().sum()

"""## String columns"""

object_columns = data.select_dtypes(include=['object']).columns
for col in object_columns:
    data[col] = data[col].str.lower()
    data[col] = data[col].fillna(data[col].mode)
data

data['total_numberofrooms']=data['bathrooms']+data['bedrooms']

data

"""## Encoding"""

dummy_cols=['has_photo','pets_allowed','price_type','source']
data=pd.get_dummies(data,columns=dummy_cols,dtype=int)

columns_to_check = ['bathrooms','bedrooms','total_numberofrooms']
for col in columns_to_check:

    data[col] = data[col].apply(lambda x: int(x) if pd.notnull(x) else None)
    data[col].astype(int)

data['price_display'] = data['price_display'].apply(lambda x: re.sub(r'\D', '', x))
data['price_display'] = data['price_display'].astype(int)

label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:

     label_encoders[col] = LabelEncoder()
     data[col] = label_encoders[col].fit_transform(data[col].astype(str))

columns_to_drop = ['price','currency'	,'fee','category']
data = data.drop(columns_to_drop, axis=1)

data

colu= data.corr()

colu

specific_column = 'price_display'


correlation_with_specific_column = data.corrwith(data[specific_column])


columns_to_drop = correlation_with_specific_column[correlation_with_specific_column.abs() <= 0.1].index


data = data.drop(columns=columns_to_drop)

x=data.drop('price_display',axis=1)
y=data['price_display']
data=pd.DataFrame(x)
data

from scipy.stats import boxcox


y, _ = boxcox(y)

plt.hist(y, bins=30, color='blue', edgecolor='black')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of latitude')
# exact normal

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)
x=scaler.transform(x)
print (x)

"""## Model"""

X_train_val, X_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=0)


rf_regressor = RandomForestRegressor(max_depth=6,n_estimators=50,max_features=3)

rf_regressor.fit(X_train, y_train)


y_train_pred = rf_regressor.predict(X_train)
y_val_pred = rf_regressor.predict(X_val)
y_test_pred = rf_regressor.predict(X_test)


mse_train = mean_squared_error(y_train, y_train_pred)
mse_val = mean_squared_error(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
r2_test = r2_score(y_test, y_test_pred)

cv_scores = cross_val_score(rf_regressor, X_train_val, y_train_val, cv=5, scoring='neg_mean_squared_error')
cv_mse_mean = -cv_scores.mean()

# Print results
print("Mean Squared Error (Train):", mse_train)
print("Mean Squared Error (Validation):", mse_val)
print("Mean Squared Error (Test):", mse_test)

print("R-squared Score (Train):", r2_train)
print("R-squared Score (Validation):", r2_val)
print("R-squared Score (Test):", r2_test)

print("Cross-Validation Mean Squared Error:", cv_mse_mean)

X_train_val, X_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=0)

xgb_regressor = xgb.XGBRegressor(alpha=0.3)


xgb_regressor.fit(X_train, y_train)

y_train_pred = xgb_regressor.predict(X_train)
y_val_pred = xgb_regressor.predict(X_val)
y_test_pred = xgb_regressor.predict(X_test)


mse_train = mean_squared_error(y_train, y_train_pred)
mse_val = mean_squared_error(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
r2_test = r2_score(y_test, y_test_pred)


cv_scores = cross_val_score(xgb_regressor, X_train_val, y_train_val, cv=5, scoring='neg_mean_squared_error')
cv_mse_mean = -cv_scores.mean()

# Print results
print("Mean Squared Error (Train):", mse_train)
print("Mean Squared Error (Validation):", mse_val)
print("Mean Squared Error (Test):", mse_test)

print("R-squared Score (Train):", r2_train)
print("R-squared Score (Validation):", r2_val)
print("R-squared Score (Test):", r2_test)

print("Cross-Validation Mean Squared Error:", cv_mse_mean)

X_train_val, X_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=0)


linear_regressor = LinearRegression()


linear_regressor.fit(X_train, y_train)


y_train_pred = linear_regressor.predict(X_train)
y_val_pred = linear_regressor.predict(X_val)
y_test_pred = linear_regressor.predict(X_test)


mse_train = mean_squared_error(y_train, y_train_pred)
mse_val = mean_squared_error(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
r2_test = r2_score(y_test, y_test_pred)


cv_scores = cross_val_score(linear_regressor, X_train_val, y_train_val, cv=5, scoring='neg_mean_squared_error')
cv_mse_mean = -cv_scores.mean()

print("Mean Squared Error (Train):", mse_train)
print("Mean Squared Error (Validation):", mse_val)
print("Mean Squared Error (Test):", mse_test)

print("R-squared Score (Train):", r2_train)
print("R-squared Score (Validation):", r2_val)
print("R-squared Score (Test):", r2_test)

print("Cross-Validation Mean Squared Error:", cv_mse_mean)

# Split the data into training, validation, and testing sets
X_train_val, X_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=0)

# Apply polynomial features
degree = 4 # Degree of the polynomial features
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_val_poly = poly_features.transform(X_val)
X_test_poly = poly_features.transform(X_test)

# Scale the features
scaler = StandardScaler()
X_train_poly_scaled = scaler.fit_transform(X_train_poly)
X_val_poly_scaled = scaler.transform(X_val_poly)
X_test_poly_scaled = scaler.transform(X_test_poly)


alpha = 0.0
ridge_regressor = Ridge(alpha=alpha)


ridge_regressor.fit(X_train_poly_scaled, y_train)


y_train_pred = ridge_regressor.predict(X_train_poly_scaled)
y_val_pred = ridge_regressor.predict(X_val_poly_scaled)
y_test_pred = ridge_regressor.predict(X_test_poly_scaled)


mse_train = mean_squared_error(y_train, y_train_pred)
mse_val = mean_squared_error(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

r2_train = r2_score(y_train, y_train_pred)
r2_val = r2_score(y_val, y_val_pred)
r2_test = r2_score(y_test, y_test_pred)

# Perform cross-validation
cv_scores = cross_val_score(ridge_regressor, X_train_val, y_train_val, cv=5, scoring='neg_mean_squared_error')
cv_mse_mean = -cv_scores.mean()

# Print results
print("Mean Squared Error (Train):", mse_train)
print("Mean Squared Error (Validation):", mse_val)
print("Mean Squared Error (Test):", mse_test)

print("R-squared Score (Train):", r2_train)
print("R-squared Score (Validation):", r2_val)
print("R-squared Score (Test):", r2_test)

print("Cross-Validation Mean Squared Error:", cv_mse_mean)