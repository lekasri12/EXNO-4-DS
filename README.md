# EXNO:4-DS
### NAME : G LEKASRI
### REGISTER NUMBER : 212223100025
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```
![ds1](https://github.com/user-attachments/assets/157a8756-20c1-4f7d-b833-bc1e5babaa04)

```
data.isnull().sum()
```
![ds2](https://github.com/user-attachments/assets/07617d72-6a2c-4749-96b5-1aedf60f6faa)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![ds3](https://github.com/user-attachments/assets/b4201c75-d6f1-415f-8cca-e17e6d98ca34)

```
data2=data.dropna(axis=0)
data2
```
![ds4](https://github.com/user-attachments/assets/98672424-420d-4c38-b60a-a7c4060b9664)

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![ds5](https://github.com/user-attachments/assets/6d8724d5-d8c3-48fe-99e2-a54118561817)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![ds6](https://github.com/user-attachments/assets/49011eef-6c13-4412-ac5e-1cd1f3bb2e02)

```
data2
```
![ds7](https://github.com/user-attachments/assets/51701a9e-3676-4152-90ec-6db68ecbbed2)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![ds8](https://github.com/user-attachments/assets/3dcd6ba8-6612-4839-982f-76c6ce438e0f)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![ds9](https://github.com/user-attachments/assets/9222b813-a8d9-48db-b464-c8550886e22b)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![ds10](https://github.com/user-attachments/assets/51ec4c64-dd27-46f0-b90d-08a85eb14115)

```
y=new_data['SalStat'].values
print(y)
```
![ds11](https://github.com/user-attachments/assets/934fecc5-9266-44e2-a302-ac147009a712)

```
x=new_data[features].values
print(x)
```
![ds12](https://github.com/user-attachments/assets/117a37c2-3c89-4113-bf47-ff0fdfe57f42)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![ds13](https://github.com/user-attachments/assets/90e4819d-7106-4844-910a-a0635b0ca0ac)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![ds14](https://github.com/user-attachments/assets/9a937a02-9581-44bd-b91e-3a695b9c9bf4)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![ds15](https://github.com/user-attachments/assets/022ec316-3fe5-45ec-b934-c3ec14f121c7)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![ds17](https://github.com/user-attachments/assets/71cb8717-13b8-453b-8471-66255f1148bd)

```
data.shape
```
![ds18](https://github.com/user-attachments/assets/007f3476-cbff-4830-913a-35edb17b30e6)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![ds19](https://github.com/user-attachments/assets/bbac1b19-5b1f-4b70-b85b-8c4603f75a80)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![ds20](https://github.com/user-attachments/assets/1e091e35-3f6e-462b-b3b7-5a1553b9eb2a)

```
tips.time.unique()
```
![ds21](https://github.com/user-attachments/assets/942c2f38-a9ca-4984-b9b0-56412ea3a9cb)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![ds22](https://github.com/user-attachments/assets/832db8e4-d803-4afd-a74b-b362e69a9287)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![ds23](https://github.com/user-attachments/assets/b7ffe95a-9b8e-4964-932a-34cf2b44478b)


# RESULT:
      
Thus the code to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is implemented successfully.
