# 1. Define Problem.
# 2. Prepare Data.
# 3. Evaluate Algorithms.
# 4. Improve Results.
# 5. Present Results.

# # Python version
# import sys
# print('Python: {}'.format(sys.version))
# # scipy
# import scipy
# print('scipy: {}'.format(scipy.__version__))
# # numpy
# import numpy
# print('numpy: {}'.format(numpy.__version__))
# # matplotlib
# import matplotlib
# print('matplotlib: {}'.format(matplotlib.__version__))
# # pandas
# import pandas
# print('pandas: {}'.format(pandas.__version__))
# # scikit-learn
# import sklearn
# print('sklearn: {}'.format(sklearn.__version__))

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#shape
# We can get a quick idea of
# how many instances (rows) and
# how many attributes (columns) the data contains
# with the shape property.
print(dataset.shape)

#head
# print(dataset.head(150))

#descriptions include the count, mean, the min and max values as well as some percentiles.
print(dataset.describe())

print()
# class distribution : look at the number of instances(rows) that belong to each class.
# print(dataset.groupby('class').size())


#Data Visualization
#1. Univariate(일변량화) plots to better understand each attribute.
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#histograms
dataset.hist()
plt.show()

#2. Multivariate(다변량화) plots to better understand the relationships between attributes.
#scatter plot matric
scatter_matrix(dataset)
plt.show()

#Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]

validation_size = 0.20
seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
	X, Y, test_size=validation_size, random_state=seed
)

#Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
