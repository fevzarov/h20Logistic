
# coding: utf-8

# ### h2o Logistic Regression Test
# ##### Testing h2o
# 
# Reference: 
# https://aichamp.wordpress.com/2017/09/29/python-example-of-building-glm-gbm-and-random-forest-binomial-model-with-h2o/

# In[1]:

# import
import h2o
h2o.init()


# In[30]:

### Data Reading

df = h2o.import_file("//txalle2cdfile25.itservices.sbc.com/APEXANALYTICS/Fahzy/prostate.csv")
df.dim
# df.summary() 
# df.col_names


# In[10]:

# Setting up predictor variable set and response variable
y = 'CAPSULE'
x = df.col_names
x.remove(y)
print("Response = " + y)
print("Pridictors = " + str(x))


# In[11]:

# Setting up a categorical variable
df['CAPSULE'] = df['CAPSULE'].asfactor()


# In[12]:

# Testing CAPSULE levels
df['CAPSULE'].levels()


# In[13]:

# Splitting dataset into train and test
train, valid, test = df.split_frame(ratios=[.8, .1])
print(df.shape)
print(train.shape)
print(valid.shape)
print(test.shape)


# In[14]:

# Logistic Regression Run
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

glm_logistic = H2OGeneralizedLinearEstimator(family = "binomial")
glm_logistic.train(x=x, y= y, training_frame=train, validation_frame=valid, 
 model_id="glm_logistic")


# In[15]:

# Extracting model's metrics
glm_logistic.varimp()


# In[16]:

# Coefficient Estimates
glm_logistic.coef()


# In[17]:

# prediction on testing dataset:

glm_logistic.predict(test_data=test)


# In[18]:

# checking the model performance metrics “rmse” based on testing and other datasets:

print(glm_logistic.model_performance(test_data=test).rmse())
print(glm_logistic.model_performance(test_data=valid).rmse())
print(glm_logistic.model_performance(test_data=train).rmse())


# In[20]:

# checking the model performance metrics “r2” based on testing and other datasets:

print(glm_logistic.model_performance(test_data=test).r2())
print(glm_logistic.model_performance(test_data=valid).r2())
print(glm_logistic.model_performance(test_data=train).r2())


# In[21]:

# Gradient Boosting Model:

from h2o.estimators.gbm import H2OGradientBoostingEstimator
gbm = H2OGradientBoostingEstimator()
gbm.train(x=x, y =y, training_frame=train, validation_frame=valid)


# In[22]:

###  MODEL PERFORMANCE ###
# confusion metrics:

gbm.confusion_matrix()


# In[23]:

# variable importance plots:

gbm.varimp_plot()


# In[24]:

# variable importance table:

gbm.varimp()


# In[25]:

# Distributed Random Forest model:

from h2o.estimators.random_forest import H2ORandomForestEstimator
drf = H2ORandomForestEstimator()
drf.train(x=x, y = y, training_frame=train, validation_frame=valid)


# In[26]:

# random forest model's 'confusion metrics:

drf.confusion_matrix()


# In[27]:

# gains and lift table:

drf.gains_lift()

