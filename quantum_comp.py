import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import random_projection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
# ---------------------------------------------------------------
train=pd.read_csv("train.csv")
print(type(train))
# ---------------------------------------------------------------
train.head()
# ---------------------------------------------------------------
train.shape
# ---------------------------------------------------------------
train.describe()
# ---------------------------------------------------------------
train.isnull().sum()
# ---------------------------------------------------------------
train.fillna(0,inplace=True)
# ---------------------------------------------------------------
y = train['label']
print(y)
x = train.drop("label",axis=1)
# ---------------------------------------------------------------
x.iloc[0].values.reshape(28,28)
type(x.iloc[0].values.reshape(28,28)
)
# ---------------------------------------------------------------
def show_images(num_images):
    for n in range(0,num_images):
        plt.subplot(5,5,n+1)
        print(type( x.iloc[n].values.reshape(28,28)))
#         plt.imshow(x.iloc[n].values.reshape(28,28))#cmap means how the imgae to be presented
        plt.xticks([]) #removes numbered labels on x-axis
        plt.yticks([])
# ---------------------------------------------------------------
show_images(25)
# ---------------------------------------------------------------
y.unique()
# ---------------------------------------------------------------
#def show_images by digit(digit):
digit=1
if digit in list(range(10)):
    indices=np.where(y==digit)#extract indecies where y==1
    for d in range(0,50):
        plt.subplot(5,10,d+1)
        data=x.iloc[indices[0][d]].values.reshape(28,28)
        plt.imshow(data)
        plt.xticks([])
        plt.yticks([])
else:
    print("number doesn't exist")
# ---------------------------------------------------------------
def fit_random_forest_classifier_with_plot(X, y):

    #First let's create training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    #We could grid search and tune, but let's just fit a simple model to see how it does
    #instantiate
    clf = RandomForestClassifier(n_estimators=100, max_depth=None)

    #fit
    clf.fit(X_train, y_train)

    #predict
    y_preds = clf.predict(X_test)

    #score
    mat = confusion_matrix(y_test, y_preds)
    print(mat)
 #   print(sns.heatmap(mat, annot=True, cmap='bwr', linewidths=.5))
    acc = accuracy_score(y_test, y_preds)
    print(acc)
    return acc
    
fit_random_forest_classifier_with_plot(x, y)
# ---------------------------------------------------------------
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection
# ---------------------------------------------------------------
rp=SparseRandomProjection(eps=0.5)#n_components='auto' or eps=''
# ---------------------------------------------------------------
x_rp=rp.fit_transform(x)
# ---------------------------------------------------------------
x
# ---------------------------------------------------------------
x_rp.shape
# ---------------------------------------------------------------
fit_random_forest_classifier_with_plot(x_rp, y)
# ---------------------------------------------------------------
  for ep in np.arange(0.5,1,0.2):
    rp=SparseRandomProjection(eps=ep)
    x_rp=rp.fit_transform(x)
    acc=fit_random_forest_classifier_with_plot(x_rp, y)
    print("With epsilon = {:.2f}, the transformed data has {} components, a random forest acheived an accuracy of {}.".format(ep, x_rp.shape[1], acc))
# ---------------------------------------------------------------
from sklearn.random_projection import johnson_lindenstrauss_min_dim
# ---------------------------------------------------------------
# Calulate the number of components with varying eps
eps=np.arange(0.1,1,0.01)
n_comp = johnson_lindenstrauss_min_dim(n_samples=1e6, eps=eps)

plt.plot(eps, n_comp, 'bo');
plt.xlabel('eps');
plt.ylabel('Number of Components');
plt.title('Number of Components by eps');
# ---------------------------------------------------------------
x_samples,x_comps=x.shape
print("The orignial data has {} samples with dimension {}.".format(x_samples, x_comps))
# ---------------------------------------------------------------
n_components = 30

rp = SparseRandomProjection(n_components=n_components)
x_rp = rp.fit_transform(x)
