import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/arunpal/Desktop/ML_PROJECT/DATASET/diabetes_prediction_dataset.csv")

gender = LabelEncoder()
smoking_history = LabelEncoder()

df['gender'] = gender.fit_transform(df['gender'])
df['smoking_history'] = smoking_history.fit_transform(df['smoking_history'])
print(df.head())

x = df[['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']]
y = df['diabetes']

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=40)

model = GaussianNB()
model.fit(xtrain,ytrain)
pred = model.predict(xtest)
print('predicted is:',pred)

print('accuracy is:',accuracy_score(ytest,pred))
print('confusion is:',confusion_matrix(ytest,pred))
print('classification report is:',classification_report(ytest,pred))

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
        'gender':[gender],
        'age':[age],
        'hypertension':[hypertension],
        'heart_disease':[heart_disease],
        'smoking_history':[smoking_history],
        'bmi':[bmi],
        'HbA1c_level':[HbA1c_level],
        'blood_glucose_level':[blood_glucose_level] 
    }
)

result = model.predict(new_data)
print('final result is:',result)

