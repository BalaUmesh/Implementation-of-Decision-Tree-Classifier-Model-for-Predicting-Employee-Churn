# Ex.no:5 Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
date: 21/9/23
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values from dataframe and apply label encoder.
3. Apply decision tree classifier on the dataframe.
4. Obtain the value of accuracy and data prediction.

## Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Bala Umesh
RegisterNumber:  212221040024
``` py
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
### data.head():
![image](https://github.com/BalaUmesh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113031742/b07cfa35-91ec-4a01-bbf4-6ec9655b837b)


### data.info():
![image](https://github.com/BalaUmesh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113031742/1377af38-ee3e-4e21-a94a-440cbf71e52c)


### is null() and sum():
![image](https://github.com/BalaUmesh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113031742/7d4bfbb0-a862-4927-8589-80d1eb9ca6d7)


### data value counts():
![image](https://github.com/BalaUmesh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113031742/4647c9c3-a294-4b25-bcfd-e6b992be52a6)


### data.head for salary:
![image](https://github.com/BalaUmesh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113031742/75340aa2-0855-4ac9-8530-0a53154134f9)


### x.head():
![image](https://github.com/BalaUmesh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113031742/981c00a7-7ab9-4b81-83c6-c8b441685eba)


### accuracy value:
![image](https://github.com/BalaUmesh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113031742/8c70a87b-43fd-4742-a510-edd1112a17d0)


### data prediction:
![image](https://github.com/BalaUmesh/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113031742/c4e2023c-2688-4b1d-9dd4-ddd6670b2297)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
