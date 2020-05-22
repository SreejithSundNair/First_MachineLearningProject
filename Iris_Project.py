import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Data import and process
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names = names)
print(dataset.head())
print(dataset.describe())
print(dataset.groupby('class').size())

# Display
dataset.plot(kind = 'box', subplots = True,  layout = (2, 2), sharex = False, sharey =  False )
pyplot.show()

# Spliting dataset in to train and test
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

models =[]
models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class = 'ovr') ))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC(gamma =  'auto')))
names =[]
model_results =[]

for name , model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_result = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    model_results.append(cv_result)
    names.append(name)
    print('%s: %f (%f)' %  (name, cv_result.mean(), cv_result.std()))

# Compare Algorithms
pyplot.boxplot(model_results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Got better accuracy models LDA with an accuracy of 97.5
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)
Prediction = model.predict(X_test)
# Model performance
print(accuracy_score(y_test, Prediction))
print(confusion_matrix(y_test, Prediction))
print(classification_report(y_test, Prediction))