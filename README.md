# Churn_Prediction
The file contains both Code and dataset.
For further queries mail @jaiprathapgv@gmail.com
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

df=pd.read_csv('Churn Modeling.csv')
df

[df.isna().any()]

Inputs=df.drop(['RowNumber','CustomerId','Surname'],axis=1)
Inputs

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Inputs['sex']=le.fit_transform(Inputs['Gender'])
print((le.classes_))
Inputs['location']=le.fit_transform(Inputs['Geography'])
print(le.classes_)

X=Inputs.drop(['Gender','Geography','Exited'],axis=1)
Y=Inputs['Exited']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,Y_train)

pre=model.predict(X_test)

model.score(X_test,Y_test)

from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

print(n_scores)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,pre)

import seaborn as sns
sns.heatmap(cm,annot=True)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Softmax, Dense,SimpleRNN,Flatten,RNN,MaxPool2D,ReLU,Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import categorical_crossentropy

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

classifier=Sequential([
    Dense(10,input_shape=(10,),activation='relu'),Flatten(),
    Dense(20,activation='relu'),
    Dense(2,activation='softmax')
])
classifier.summary()

model1=Sequential()
model1.add(Dense(10,activation='relu'))
model1.add(Dense(5,activation='relu'))
model1.add(Dense(2,activation='softmax'))


classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=20)

