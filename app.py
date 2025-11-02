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
#print(cat_cols)
cat_cols=pd.get_dummies(df[cat_cols],dtype=int,drop_first=True)
#num_cols=pd.DataFrame(num_cols)
fur_encoded=pd.get_dummies(df['furnishingstatus'],dtype=int)
df1=pd.concat([cat_cols,fur_encoded],axis=0)
#df2=pd.concat([df1,num_cols],axis=0)
print(df1.info())