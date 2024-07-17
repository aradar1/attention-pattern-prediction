import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
path = "SampleData/clustered nostudentid .csv"

headernames = ['AVTYPE', 'Ethnicity', 'Gender','Age','A.Delta','A.Theta','A.Alpha1','A.Alpha2','A.Beta1','A.Beta2','A.Gamma1','A.Gamma2','A. Attention','A. Mediation','Class']

dataset = pd.read_csv(path, names=headernames)
dataset.head() 

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 14].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=60)
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)





