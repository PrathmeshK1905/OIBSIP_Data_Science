#!/usr/bin/env python
# coding: utf-8

# # Name: Prathmesh Kulkarni

# # Car Price Prediction with Machine Learning 

# Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# Importing Dataset

# In[3]:


data=pd.read_csv('https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv')


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.shape


# In[7]:


print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])


# In[8]:


data.info()


# In[9]:


data.isnull().sum()


# In[10]:


data = data.dropna()


# In[11]:


data = data.drop_duplicates()


# Visualising Data

# In[12]:


plt.style.use('dark_background')
sns.set_palette('dark')
sns.histplot(data['price'])
plt.title('Distribution of Car Prices',color ='white')
plt.xlabel('Price',color ='white')
plt.ylabel('Count',color ='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()

numeric_features = ['wheelbase','carlength','carwidth','carheight','curbweight',
                    'enginesize','boreratio','stroke','compressionratio','horsepower',
                    'peakrpm','citympg','highwaympg','price']
correlation_matrix = data[numeric_features].corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot =True, cmap='coolwarm')
plt.title('Correlation Heatmap',color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()


# In[13]:


feature_cols = ['symboling','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation',
               'wheelbase','carlength','carwidth','carheight','curbweight','enginetype','cylindernumber',
               'enginesize','fuelsystem','boreratio','stroke','compressionratio','horsepower','peakrpm',
               'citympg','highwaympg']
target_col ='price'
x = data[feature_cols]
y = data[target_col]


# In[14]:


label_encoder = LabelEncoder()
for col in x.columns:
    if x[col].dtype == 'object':
        x[col] = label_encoder.fit_transform(x[col])


# Splitting Dataset into Test and Train

# In[15]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)


# Training Model

# In[16]:


model = LinearRegression()
model.fit(x_train,y_train)


# In[17]:


predictions = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:",rmse)


# Testing Prediction

# In[18]:


new_car_data = [[3,'diesel','std','four','convertible','fwd','front',
                 100.0,180.0,65.0,56.0,2500,'ohc','four',120,'mpfi',
                 3.50,2.80,8.0,120,5500,30,38]]
new_car_df = pd.DataFrame(new_car_data,columns=feature_cols)
new_car_encoded = pd.get_dummies(new_car_df,drop_first = True)
new_car_encoded = new_car_encoded.reindex(columns=x_train.columns, fill_value=0)
predicted_price = model.predict(new_car_encoded)
print("Predicted Price:",predicted_price)


# END
