import pandas as pd

dfpenguins = pd.read_csv('Penguins.csv')

target = 'species'#variable a predecir
encode =['sex','island']

for col in encode:
    dummy = pd.get_dummies(dfpenguins[col], prefix=col)
    dfpenguins = pd.concat([dfpenguins,dummy],axis=1)
    del dfpenguins[col]

target_mapper = {'Adelie':0, 'Chinstrap':1,'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

dfpenguins['species'] = dfpenguins['species'].apply(target_encode)

#Separar x y y
X = dfpenguins.drop('species', axis=1)#input features
Y = dfpenguins['species']#species

#Random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X,Y)

#Guardar el modelo
import pickle
pickle.dump(clf,open('penguins_clf.pkl','wb'))