'''Citation for the database
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing.
 Decision Support Systems, Elsevier, 62:22-31, June 2014
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

## Data Preprocessing
dataset = pd.read_csv('bank-additional-full.csv', sep=';')
# Onehot Encoding categorical variables
Cat_Columns = [col for col in dataset.columns if dataset[col].dtype == object]
dataset_Encode = pd.get_dummies(dataset, columns=Cat_Columns, drop_first=True)
X = dataset_Encode.iloc[:, :-1].values  # Independent variable
y = dataset_Encode.iloc[:, -1].values   # encoded target variable
# Train test data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

models =[]
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('NB', GaussianNB()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVC',SVC()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))

names = []
model_results = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_result = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    model_results.append(cv_result)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_result.mean(), cv_result.std()))

# two models LDA and LogisticRegression better accurate model
model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, y_train)
prediction = model.predict(X_test)

print(confusion_matrix(y_test, prediction))
print(accuracy_score(y_test, prediction))
print(classification_report(y_test,prediction))