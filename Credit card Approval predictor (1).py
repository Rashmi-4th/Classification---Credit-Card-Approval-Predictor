#!/usr/bin/env python
# coding: utf-8

# In[12]:


#### IMPORTING THE DATASET AND THE PACKAGES
 
import os
os.getcwd()
os.chdir(r'C:\Users\Rashmi\Desktop\Projects\Project-1\Project')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


##LOAD DATASET

cc_app=pd.read_csv('crx.data', header = None)
cc_app


# In[15]:


## DATA INSPECTION

print(cc_app.describe)


# In[16]:


## DATAFRAME INFORMATION

cc_app.info()


# In[18]:


##MISSING VALUE INSPECTION IN DATASET

print(cc_app.isnull().sum())


## REPLACING '?'s' with nan
cc_app = cc_app.replace('?', np.NaN)
cc_app


# In[6]:


## Inspecting the missing value again

print('Total NaN: '+ str(cc_app.isnull().values.sum()))
print('NaN by column: ' '\n')
print(cc_app.isnull().sum())


# In[10]:


## HANDLING THE MISSING VALUES USING MEAN

cc_app.fillna(cc_app.mean(), inplace=True)
cc_app
print('Total NaN: '+ str(cc_app.isnull().values.sum()))
print('NaN by column: ' '\n')
print(cc_app.isnull().sum())


# In[55]:


## ITERATE OVER EACH COLUMN OF CC_APP

for col in cc_app:
    ## CHECK IF THE COLUMN IS OF OBJECT TYPE
    if cc_app[col].dtypes == 'object':
        ## IMPUTE WITH THE MOST FREQUENT VALUES
        cc_app = cc_app.fillna(cc_app[col].value_counts().index[0])
##COUNT THE NO. OF NANs AND PRINT THE COUNT TO VERIFY
print('Total NaN: '+ str(cc_app.isnull().values.sum()))
print('NaN by column: ' '\n')
print(cc_app.isnull().sum())        


# In[56]:


## IMPORTING LABELENCODER

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

## ITERATE OVER ALL THE VALUES OF EACH COLUMN AND GET THEIR DTYPES
for col in cc_app:
    ## COMPARE IF DTYPE IS OBJECT
    if cc_app[col].dtypes == 'object':
        ## USE LABELENCODER TO DO NUMERICAL TRANSFORMATION
        le.fit(cc_app[col])
        cc_app[col]=le.transform(cc_app[col])
        
cc_app.info()        


# In[57]:


cc_app


# In[58]:


## SCALING THE FEATURE VALUES TO AN UNIFORM RANGE

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_std = cc_app.iloc[:,1:15]
x_std = sc_x.fit_transform(x_std)
x = x_std


# In[15]:


x


# In[16]:


## SPLITTING THE DATASET INTO TRAIN TEST SETSA

from sklearn.model_selection import train_test_split

## SEGREGATE FEATURES AND LABELS INTO SEPARATE VARIABLES
x=cc_app.iloc[:, 0:15].values
y=cc_app.iloc[:,-1].values


# In[17]:


x


# In[18]:


y


# In[19]:



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=10)


# In[20]:


x_test


# In[21]:


x_train


# In[22]:


y_test


# In[23]:


y_train


# In[65]:


## IMPORTING LOGISTIC REGRESSION AND FITTING THE MODEL TO THE TRAIN SET

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)


# In[39]:



y_pred = logreg.predict(x_test)
y_pred


# In[36]:


## IMPORT CONFUSION MATRIX AND ACCURACY SCORE

from sklearn.metrics import confusion_matrix, accuracy_score
## USE THE MODEL TO PREDICT INSTANCES FROM THE TEST SET AND STORE IT

y_pred = logreg.predict(x_test)
print(logreg.score(x_test, y_test))

## GET THE ACCURACY SCORE OF THE MODEL 
print(accuracy_score(y_test,y_pred))


# In[35]:


confusion_matrix(y_test, y_pred)


# In[52]:


## IMPORT GRID SEARCH AND MAKING THE MODEL PERFORM BETTER

from sklearn.model_selection import GridSearchCV
tol = [0.1,0.01,0.001,0.0001]
max_iter = [100,160,150,200,]
param_grid = dict(tol=tol, max_iter=max_iter)
grid_model = GridSearchCV(estimator = logreg, param_grid = param_grid, cv=5) 


# In[59]:


x = sc_x.fit_transform(x)


# In[63]:


grid_model_result = grid_model.fit(x,y)
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print(best_score, best_params)


# In[ ]:




