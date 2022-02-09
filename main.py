import pandas as pd
import six
import numpy as np
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.decomposition import PCA
from matplotlib.pyplot import figure
from sklearn.ensemble import AdaBoostClassifier

df4 = pd.read_csv(r"D:\DataAnalysis\01_assestament\breast-cancer.csv")
#checking variables type to understand best process
print(df4.dtypes)
#checking numer of null values
print('nulls values in database')
print(df4.isnull().sum())
#checking tail of the dataset to calculate ideal test enviroment
print(df4.tail())
#checking total of class and values of the column "diagnosis"
print (df4['diagnosis'].value_counts())
#relation between diagnosis and values
#creating graphs features
plt.figure(figsize=(10,18), facecolor='white')
plotnumber = 1

for column in df4:
    if plotnumber<=31 and column!='diagnosis' :
        ax = plt.subplot(8,4,plotnumber)
        sns.histplot(df4[column])
        plt.xlabel(column,fontsize=5)
        plt.ylabel('values', fontsize=5)
    plotnumber+=1
print(plt.show())

#checking outliners
fig, ax = plt.subplots(figsize=(15,10))
print(sns.boxplot(data=df4, width= 0.5,ax=ax,  fliersize=3))

#splitting dataset into input features and output feature
X = df4.drop(columns = 'diagnosis')
y = df4['diagnosis']
#creating training and test sets
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.30, random_state= 1200456788)
#creating "original" tree model
#let's first visualize the tree on the data without doing any pre processing
clf = DecisionTreeClassifier()
#feeding train dataset to the tree model, with .fit comand the system is deciding the splitting citeria used
#and the root node to create the decision path
clf.fit(x_train,y_train)
input_name=list(X.columns)
class_name = list(y_train.unique())
#taking classifier values and features names to draw the tree model
# create a dot_file which stores the tree structure
dot_data = export_graphviz(clf,feature_names = input_name,rounded = True,filled = True)
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("model1.png")
# Show graph print(Image(graph.create_png()))
#test our tree accuracy train and test dataset
print('training pre PCA',clf.score(x_train,y_train))
print('test pre PCA',clf.score(x_test,y_test))
py_pred = clf.predict(x_test)
#print(py_pred)
#scaling data to make the dataset easier to understand for the model
scalar = StandardScaler()
x_transform = scalar.fit_transform(X)
#creating training enviroment at 20%
x_train,x_test,y_train,y_test = train_test_split(x_transform,y,test_size = 0.20, random_state= 1200456788)
#checking if with PCA the model can become more accurate, I am looking for the slope to understand out of all the columns
#which is the efficient number for the model even if the first analysis shows 92% accuracy

pca = PCA()
principalComponents = pca.fit_transform(x_transform)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Explained Variance')
plt.show()
#around 95% of the variance is explained by 10 variables
pca = PCA(n_components=10)
new_data = pca.fit_transform(x_transform)
principal_x = pd.DataFrame(new_data,columns=['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8','PC-9','PC-10'])
# let's see how well our model perform on this new data
x_train,x_test,y_train,y_test = train_test_split(principal_x,y,test_size = 0.20, random_state= 1200456788)
#let's first visualize the tree on the data without doing any pre processing
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
print('training after PCA',clf.score(x_train,y_train))
print('test after PCA',clf.score(x_test,y_test))
print('increasing the randomness in the splitting of the dataset the PCA results in a higher efficiency in terms of prediction while increasing the test size will reduce it')

#checking hyperparameters to decide what are the optimum values to determine the best output
#using adaptive boosting to check the accuracy of the dataset
# Instantiate a normalized linear regression model
reg_lm = DecisionTreeClassifier()

# Build and fit an AdaBoost regressor
reg_ada = AdaBoostClassifier()
reg_ada.fit(x_train, y_train)

# Calculate the predictions on the test set
pred = reg_ada.predict(x_test)
print(accuracy_score(y_test, pred))
