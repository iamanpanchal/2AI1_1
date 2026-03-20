import pandas as pd
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv('insurance_data_linear.csv')
df.head()
df.info()
le=LabelEncoder()
#This creates a tool that converts labels - numbers

df['sex']=le.fit_transform(df['sex'])
df['smoker']=le.fit_transform(df['smoker'])
df['region']=le.fit_transform(df['region'])

# fit_transform(): learns categories and converts them into numbers

print(df.head())
