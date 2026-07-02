import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/Users/arunpal/Desktop/ML_PROJECT/DATASET/heart.csv")
print(df.head())

x = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y = df['target']

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(xtrain,ytrain)
pred = model.predict(xtest)
print('predicted:',pred)

print('accuracy:',accuracy_score(ytest,pred))
print('confusion:',confusion_matrix(ytest,pred))
print('classification:',classification_report(ytest,pred))

age = int(input('enter ur age:'))
sex = int(input('enter ur sex:'))
cp = int(input('enter ur cp:'))
trestbps = int(input('enter ur trestbps:'))
chol = int(input('enter ur chol:'))
fbs = int(input('enter ur fbs:'))
restecg = int(input('enter ur restecg:'))
thalach = int(input('enter ur thalach:'))
exang = int(input('enter ur exang:'))
oldpeak = float(input('enter ur oldpeak:'))
slope = float(input('enter ur slope:'))
ca = int(input('enter ur ca:'))
thal = int(input('enter ur thal:'))

new_data = pd.DataFrame(
    {
        'age': [age],       
        'sex': [sex],    
        'cp':  [cp],        
        'trestbps':[trestbps],    
        'chol':   [chol],     
        'fbs':   [fbs],      
        'restecg':  [restecg],   
        'thalach':  [thalach],   
        'exang':   [exang],    
        'oldpeak': [oldpeak],
        'slope':  [slope],
        'ca':  [ca],
        'thal':  [thal]
    }
)

result = model.predict(scaler.fit_transform(new_data))
print('final result is:',result)