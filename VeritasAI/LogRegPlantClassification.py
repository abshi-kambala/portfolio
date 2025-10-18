# import libraries and files
# please do not change this code
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# load data library
# please do not change this code
from sklearn.datasets import load_iris

# get data using function imported above
# please do not change this code
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target

# do not change this code
df.head(3)

# define explanatory variables as X
## Your Code Starts Here ##
X = df.iloc[:, :4]

# print the shape of X
## Your Code Starts Here ##
np.shape(X)

# define target variable as Y
## Your Code Starts Here ##
Y = df['species']

# please do not change this code
from sklearn.model_selection import train_test_split

# separate training and test set data
## Your Code Starts Here ##
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# please do not change this code
# ensure data is in the numpy array
# please do not change this code
# you have to run this code because
# the scripts downstream depends on this
# code
X_train = np.array(X_train)
y_train = np.array(y_train).reshape(-1, 1)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

lr = LogisticRegression()
lr.fit(X_train, Y_train)
lr.coef_

results = lr.predict(X_test)
np.mean(results == Y_test)

# compute test set confusion table
## Your Code Starts Here ##
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(Y_test, results)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

'''

# instead of predictions, get predicted probabilities of breast cancer
y_test_proba = logit_model.predict_proba(X_test)[:,1]
y_test_proba 

results = y_test_proba > 0.1
results

results = np.multiply(y_test_pred, 1)
results


'''
