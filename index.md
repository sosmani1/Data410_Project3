# Advanced Machine Learning Project 3

# *DEFINITIONS*

# *Multivariate Regression Analysis:* 
Multivariate Regression Analysis is when we want to predict a target variable using more than one feature/input variable.

As explained by Brillant.org  Multivariate Regression is a method used to measure the degree at which more than one independent variable (predictors) and more than one dependent variable (responses), are linearly related. The method is broadly used to predict the behavior of the response variables associated to changes in the predictor variables, once a desired degree of relation has been established.

![0_AqzOn7p--nveVULA](https://user-images.githubusercontent.com/78623027/155774092-b897fd7e-f5e8-455b-91e8-dbd183859f50.png)

# *Gradient Boosting:* 
Gradient boosting is a type of machine learning technique. The main idea is to set the target variable for the next decision tree model using predictions from the previous models in order to minimize the error and create a better model.

Gradient boosting involves three elements:

* A loss function to be optimized.
* A weak learner to make predictions.
* An additive model to add weak learners to minimize the loss function.


The following image from Akira-AI shows a visual interpration of Gradient Boosting: 

![akira-ai-gradient-boosting-ml-technique](https://user-images.githubusercontent.com/78623027/155774282-b4c9d364-ccdb-4fa1-a80c-6b661bc49fd4.png)


# *Extreme Gradient Boosting:* 

Extreme Gradient Boosting is similar to gradient boosting, but makes use of regularization parameters, to prevent overfitting.

According to MachinelearningMastery.com Gradient boosting refers to a class of ensemble machine learning algorithms that can be used for classification or regression predictive modeling problems.

Ensembles are constructed from decision tree models. Trees are added one at a time to the ensemble and fit to correct the prediction errors made by prior models. This is a type of ensemble machine learning model referred to as boosting.

Models are fit using any arbitrary differentiable loss function and gradient descent optimization algorithm. This gives the technique its name, “gradient boosting,” as the loss gradient is minimized as the model is fit, much like a neural network.

The following image shows a visual of Extreme Gradient Boosting
![The-structure-of-extreme-gradient-boosting](https://user-images.githubusercontent.com/78623027/155774538-ec4823c2-0c3e-44dd-9e28-1f5b2b1782d3.png)

```
### Lets import some information as we get started. 

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error

# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 
 ``` 
 
# Defining locally weighted regression model and boosted locally weighted regression model functions:

```
#Defining the kernel local regression model

def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        #A = A + 0.001*np.eye(X1.shape[1]) # if we want L2 regularization
        #theta = linalg.solve(A, b) # A*theta = b
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      # output[np.isnan(output)] = g(X[np.isnan(output)])
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
  ```
  ```
  def boosted_lwr(X, y, xnew, kern, tau, intercept):
  # we need decision trees
  # for training the boosted method we use X and y
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  # Now train the Decision Tree on y_i - F(x_i)
  new_y = y - Fx
  #model = DecisionTreeRegressor(max_depth=2, random_state=123)
  model = RandomForestRegressor(n_estimators=100,max_depth=2)
  #model = model_xgb
  model.fit(X,new_y)
  output = model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept)
  return output 
  ```
  ```
  import xgboost as xgb
  ```
## Import data, use regularization for variable selection:
```
# import the data
housing_prices = pd.read_csv('Boston Housing Prices.csv')

train = housing_prices.drop(['cmedv', 'town', 'river'], axis = 1)
train_labels = housing_prices.cmedv

lr = Lasso(alpha=1)
lr.fit(train,train_labels)
lr.coef_
```

array([-2.86352877e-04, -0.00000000e+00,  0.00000000e+00, -7.39950402e-02,
        5.11731996e-02, -0.00000000e+00, -0.00000000e+00,  8.43070951e-01,
        2.16244065e-02, -6.82400968e-01,  2.21336096e-01, -1.64168532e-02,
       -7.02447247e-01, -7.94107775e-01])
       
```
features = np.array(train.columns)
print("All features:")
print(features)
```


All features:
['tract' 'longitude' 'latitude' 'crime' 'residential' 'industrial' 'nox'
 'rooms' 'older' 'distance' 'highway' 'tax' 'ptratio' 'lstat']
 
```
X = housing_prices[['rooms', 'crime', 'lstat']].values
y = housing_prices['cmedv'].values
scale = StandardScaler()
```

## Calculate cross-validated MSE and MAE for lowess, boosted lowess, and xgboost:

```
# we want more nested cross-validations

mse_lwr = []
mse_blwr = []
mse_xgb = []
for i in [123]:
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    yhat_lwr = lw_reg(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
    yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)
    mse_lwr.append(mse(ytest,yhat_lwr))
    mse_blwr.append(mse(ytest,yhat_blwr))
    mse_xgb.append(mse(ytest,yhat_xgb))
print('The Cross-validated Mean Squared Error for LWR is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for BLWR is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
```
* The Cross-validated Mean Squared Error for LWR is : 18.84986146382691
* The Cross-validated Mean Squared Error for BLWR is : 17.7551159425555
* The Cross-validated Mean Squared Error for XGB is : 16.71422016914427

```
mae_LWR = mean_absolute_error(ytest,yhat_lwr)
print("MAE LWR = ${:,.2f}".format(1000*mae_LWR))
mae_BLWR = mean_absolute_error(ytest,yhat_blwr)
print("MAE BLWR = ${:,.2f}".format(1000*mae_BLWR))
mae_XGB = mean_absolute_error(ytest,yhat_xgb)
print("MAE XGB = ${:,.2f}".format(1000*mae_XGB))
```
* MAE LWR = $2,659.55
* MAE BLWR = $2,651.25
* MAE XGB = $2,404.75

### XGB is the best method, because the cross-validated MSE is the lowest.
