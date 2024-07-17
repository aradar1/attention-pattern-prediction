from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score


path = "SampleData/clustered nostudentid .csv"

headernames = ['AVTYPE', 'Ethnicity', 'Gender','Age','A.Delta','A.Theta','A.Alpha1','A.Alpha2','A.Beta1','A.Beta2','A.Gamma1','A.Gamma2','A. Attention','A. Mediation','Class']
data = read_csv(path, names=headernames)
array = data.values

X = array[:,0:14]
Y = array[:,14]



seed = 7
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
cart = DecisionTreeClassifier()
num_trees = 80

model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees,random_state=seed)
model.fit(X,Y)


results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy:%0.2f " % (results.mean()))
print(results)




