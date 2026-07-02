import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

df = pd.read_csv("C:/Users/arunpal/Desktop/ML_PROJECT/DATASET/diabetes_prediction_dataset.csv")
print(df.head())
gender = LabelEncoder()
smoking_history = LabelEncoder()

df['gender'] = gender.fit_transform(df['gender'])
df['smoking_history'] = smoking_history.fit_transform(df['smoking_history'])

# print(df.head())

x = df[['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']]
y = df['diabetes']

xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=0.2,random_state=43)

scaler = StandardScaler()

xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain,ytrain)
pred = model.predict(xtest)
print('predicted is:',pred[:8])

print('accuracy:',accuracy_score(ytest,pred))
print('confusion_matrix:',confusion_matrix(ytest,pred))
print('classification_report:',classification_report(ytest,pred))

gender = int(input('enter ur gender:'))
age = int(input('enter ur age:'))
hypertension = float(input('enter ur hypertension:'))
heart_disease = float(input('enter ur disease:'))
smoking_history = float(input('enter smkhistory:'))
bmi = float(input('enter ur bmi:'))
HbA1c_level = float(input('enter ur HbA1c_level:'))
blood_glucose_level = float(input('enter ur blood_glucose_level :'))

new_data = pd.DataFrame(
    {
      'gender' : [gender],
      'age' : [age],
      'hypertension':[hypertension],
      'heart_disease':[heart_disease],
      'smoking_history':[smoking_history],
      'bmi':[bmi],
      'HbA1c_level':[ HbA1c_level],
      'blood_glucose_level':[blood_glucose_level] 
      
    }
)

result = model.predict(scaler.transform(new_data))
print('final result is:',result)