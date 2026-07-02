import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("C:/Users/arunpal/Desktop/ML_PROJECT/DATASET/diabetes_prediction_dataset.csv")
# print(df.head())

gender = LabelEncoder()
smk = LabelEncoder()

df['gender'] = gender.fit_transform(df['gender'])
df['smoking_history'] = smk.fit_transform(df['smoking_history'])

x = df[['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']]
y = df['diabetes']

print(df.head())

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier()
print(model)
model.fit(x_train,y_train)
pred = model.predict(x_test)
print('prediction is:',pred[:5])
print(df.columns)

print('accuracy:',accuracy_score(y_test,pred))
print('confusion:',confusion_matrix(y_test,pred))
print('classification report:',classification_report(y_test,pred))

gender = int(input('enter ur gender:'))
age = int(input('enter ur age:'))
hypertension = float(input('enter ur hypertension:'))
heart_disease = float(input('enter ur disease:'))
smoking_history = float(input('enter smkhistory:'))
bmi = float(input('enter ur bmi:'))
HbA1c_level = float(input('enter ur HbA1c_level:'))
blood_glucose_level = float(input('enter ur blood_glucose_level :'))


result = model.predict([[gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level]])
print('final result is:',result[:3])

prob =  model.predict_proba([[gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level]])
print('diabities negative:',prob[0][0])
print('diabities positive:',prob[0][1])

plot_tree(
    model,
    feature_names=df.columns,
    class_names=['no diabities', 'diabities'],
    filled=True
)
plt.show()