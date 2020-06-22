# Data Science Cardiovascular Disease Classification: Project Overview
* Created a tool that identifies data science cardiovascular disease (Accuracy: 81%) to help doctors notice the disease in the future.
* Used data set of Cardiovascular Disease in Kaggle (https://www.kaggle.com/sulianova/cardiovascular-disease-dataset#cardio_train.csv). 
* Optimized Logistic Regression, K Nearest-Neighbors, Naive Bayes, XGBoost, LightGBM, and Random Forest Regressors using GridSearchCV to reach the best model.
* Built a model explanation using SHAP.   

## Code and Resources Used
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, statsmodels, xgboost, lightgbm, matplotlib, seaborn, shap    

## Data Collecting
In the dataset, we got the following features:
* General information: age, weight, height, and gender
* Physical index: systolic blood pressure, diastolic blood pressure, cholesterol, and glucose
* Living habits: daily smoking, alcohol intake, and physical activities
* Target: Whether they are a cardiovascular disease patient  

## Data Cleaning
I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:
* Scaled the feature to standardization.
* Splitted the data into train and validation sets with validation size of 20%

## EDA
I looked at the normality of the data, the correlation with the various variables. Below are a few highlights from the figures.
Correlation with other features:
![alt text](https://github.com/ILing82816/ds_disease_proj/blob/master/var_corr.png "correlation")    

## Model Building  
I tried six different models and evaluated them using ROC curve. I chose ROC curve because it is relatively easy to interpret and check overfitting for this type of model.  
I tried six different models:  
* **Logistic Regression, KNN, Naive Bayes** - Baseline for the model
* **XGBoost, LightGBM** - Because of the ensemble model improving weak classifiers, I thought XGBoost and LightGBM would be effective.
* **Random Forest** - Because of larger variance of XGBoost and LightGBM, I thought random forest would be hlpful when the previous model easy to overfit.   

## Model performance
The Random Forest model have more good fit on the train and validation sets.
* **Logistic Regression:** Accuracy on train and validation sets = 78%
* **KNN:** Accuracy on train sets = 75%, Accuracy on validation sets = 72%
* **Naive Bayes:** Accuracy on train sets = 82%, Accuracy on validation sets = 78%
![alt text](https://github.com/ILing82816/ds_oil_price_proj/blob/master/Figure/prediction_prophet.png "prophet")   
* **Linear Regression:** MAE = 0.82  
![alt text](https://github.com/ILing82816/ds_oil_price_proj/blob/master/Figure/prediction_linear.png "linear")  
* **Long Short-term Memory (LSTM):** MAE = 1.08  
![alt text](https://github.com/ILing82816/ds_oil_price_proj/blob/master/Figure/prediction_LSTM.png "LSTM")

## Productionization
In this step, I built a flask API endpoint that was hosted on a local webserver by following along with the tutorial in the reference section above. The API endpoint takes in a request with the day of prediction and returns a list of estimated WTI Price.
