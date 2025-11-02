import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#EDA
df=pd.read_csv('Housing.csv')
cat_cols = [cols for cols in df.columns if df[cols].dtype == 'object']
num_cols = [cols for cols in df.columns if df[cols].dtype != 'object']
'''print("Categorical Columns:", cat_cols)
print("Numerical Columns:", num_cols)'''

#feature engineering
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder
cat_cols.remove('furnishingstatus')

# One-hot encode categorical columns
cat_encoded = pd.get_dummies(df[cat_cols], dtype=int, drop_first=True)

# One-hot encode furnishing status
fur_encoded = pd.get_dummies(df['furnishingstatus'], dtype=int)

# Scale numerical features
scaler = StandardScaler()
num_data = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

# Combine all features
df_processed = pd.concat([num_data, cat_encoded, fur_encoded], axis=1)

#print(df_processed.info())
X=df_processed[['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
       'mainroad_yes', 'guestroom_yes', 'basement_yes', 'hotwaterheating_yes',
       'airconditioning_yes', 'prefarea_yes', 'furnished', 'semi-furnished',
       'unfurnished']]
Y=df_processed[['price']]
# print(Y.columns)

#model selection 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.23,random_state=100)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

'''area_=int(input('Enter the value of area'))
bedrooms=int(input('Enter the no of bedrooms'))
bathrooms=int(input('Enter the no of bathrooms'))
stories=int(input('Enter the value of stories'))'''

Y_pred=regressor.predict(X_test)

#testing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Calculate regression metrics
r2 = r2_score(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, Y_pred)



print("\nModel Performance Metrics:")
print(f"R² Score = {r2:.4f}")  # How well the model fits (0 to 1, higher is better)
print(f"Root Mean Squared Error = {rmse:.2f}")  # Average error in same units as price
print(f"Mean Absolute Error = {mae:.2f}")  # Average absolute error in same units as price

#improving model
corr_matrix= df_processed.corr()
print(corr_matrix)
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix,annot=True)
plt.title('correlation matrix')
plt.tight_layout()
plt.savefig('correlationmatrix.jpg',dpi=300,bbox_inches='tight')
plt.show()

