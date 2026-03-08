import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, roc_auc_score
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.head()
penguins.drop(penguins[penguins['body_mass_g'].isnull()].index,axis=0, inplace=True)
penguins['sex'] = penguins['sex'].fillna('MALE')
penguins.drop(penguins[penguins['sex']=='.'].index, inplace=True)
penguins.groupby('species').mean(numeric_only =True)
df = penguins.copy()
target = 'sex'
encode = ['species','island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df.sex.unique
target_mapper = {'male':0, 'female':1,'MALE':0}
def target_encode(val):
    return target_mapper[val]
Y  =df['sex']
df['sex'] = df['sex'].apply(target_encode)
X = df.drop('sex', axis=1)
y = df['sex']
from sklearn import preprocessing
X = preprocessing.scale(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=13)
from sklearn.tree import DecisionTreeClassifier
DT1 = DecisionTreeClassifier(criterion = 'gini' , max_depth = 5 , min_samples_split = 2, random_state = 100)
DT1.fit(X_train,y_train)
ypred = DT1.predict(X_test)
y_pred_prob = DT1.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC Score: {auc_score:.3f}")
precision = precision_score(y_test, y_pred)
print(f"Precision Score: {precision:.3f}")

