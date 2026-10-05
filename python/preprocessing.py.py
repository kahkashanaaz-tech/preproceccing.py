import pandas as pd 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
df = pd.read_csv("data_preprocessing_dataset-5.csv") 
print("--- Original Dataset ---") 
print(df) 
numeric_features = ["Age", "Salary", "Years of Experience"] 
imputer = SimpleImputer(strategy="median") 
df[numeric_features] = imputer.fit_transform(df[numeric_features]) 
standard_scaler = StandardScaler() 
standard_data = standard_scaler.fit_transform(df[numeric_features]) 
minmax_scaler = MinMaxScaler() 
minmax_data = minmax_scaler.fit_transform(df[numeric_features]) 
print("\n--- StandardScaler ---") 
print(standard_data) 
print("\n--- MinMaxScaler ---") 
print(minmax_data) 
print("\n--- StandardScaler Range ---") 
print("Minimum:", standard_data.min(axis=0)) 
print("Maximum:", standard_data.max(axis=0)) 
print("\n--- MinMaxScaler Range ---") 
print("Minimum:", minmax_data.min(axis=0)) 
print("Maximum:", minmax_data.max(axis=0))