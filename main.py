import six
import graphviz
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
#importing dataset
df4 = pd.read_csv(r"D:\DataAnalysis\01_assestament\breast-cancer.csv")
print(df4)
#checking variables type to understand best process
print(df4.dtypes)
#checking number of null values (empty values)
print('Nulls values in database')
print(df4.isnull().sum())
#checking number of nan values (not a number value)
print('No number values in database')
print(df4.isna().sum())
#checking tail of the dataset to calculate ideal test enviroment
print(df4.tail())
#checking total of class and values of the column "diagnosis"
print (df4['diagnosis'].value_counts())
#dropping column ID since is not relevant for our analysis
df4 = df4.drop('id', axis = 1)
print(df4.head())
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
#removing outliners
#creating function to remove outliers from columns
def remove_outliers(x, y, LB, UB):
    LB_q = x.quantile(LB)
    UB_q = x.quantile(UB)
    mask = (x < LB_q) | (x > UB_q)
    is_outlier = x.apply(lambda row: all(row), axis=1)
    x_filt = x.loc[is_outlier].reset_index(drop=True)
    y_filt = y.loc[is_outlier].reset_index(drop=True)
    return x_filt, y_filt
#cleaning outliers
LB = 0.05
UB = 0.95
x = df4.drop(columns=['diagnosis'])
y = df4[['diagnosis']]
x_filt,y_filt = remove_outliers(x, y, LB, UB)
#printing x features after outliers removal
print(x_filt)
#printing y features after outliers removal
print(y_filt)
#number of outliers identified
original_df4 = df4.shape[0]
outliers_row = x_filt.shape[0]
outliers_removed = original_df4 - outliers_row
print('Have been removed', outliers_removed, 'outliers')
#assigning new df to run PCA
pca = PCA()
x_new = pca.fit_transform(x_filt)
print(x_new.shape)
#creating explained variance to verify importance of single Principal component
PC_values = np.arange(pca.n_components_) + 1
fig = plt.figure(figsize=(16,9))
plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Explained Variance')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.grid()
print(plt.show())
#creating cumulative variance to verify number of Principal components that we need to proper explain df4 variance
exp_var = pca.explained_variance_ratio_
fig = plt.figure(figsize=(16,9))
plt.plot(range(len(exp_var)),exp_var.cumsum(), '-ro')
plt.title('Cumulative Explained Variance')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
print (plt.grid())
#we can see that number of PC needed to explain at least 90% is just 1, slope of the variance is with 2
#using therefore slope of the graph to see features relevance for PC
pca = PCA(n_components=2)
x_new = pca.fit_transform(x_filt)
loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = x_filt.columns.values
loadings_df = loadings_df.set_index('variable')
print (loadings_df)
#using seaborn function to have a better visualization of the wight of the feature in the PC
fig = plt.figure(figsize=(16,9))
ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral', )
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 18)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 18)
print(plt.show())
#we see that in PC1 there's a strong relation within 'area worst' and 'area mean' while in PC2 their relation is negative
#creating Train,Validation & Test enviroments
x_train, x_test, y_train, y_test = train_test_split(x_filt, y_filt, test_size=0.2, random_state=1) #train set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1) #validation and test
#using Tree-based model
#validation set
#creating variable that returns tree-based model accuracy, manually assigning hyper parameters
def compute_DecisionTree_accuracy(x_train, y_train, x_val, y_val, max_depth, min_samples_split, seed):
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=seed)
    predictions = model.fit(x_train, y_train).predict(x_val)
    accuracy = accuracy_score(y_val, predictions)
    return accuracy
#running tree-based model accuracy over the hyper parameters selected
n_runs = 5
accuracy = [] #using list over dictionary because I need a position of the value more that a key/value relation
results_list = []
max_depth_list = [2, 5, 10, 15, 20, 25] #using these values because of tests executed in precedence and because of studies of resources
min_samples_split_list = [2, 5, 10, 20, 50]
for max_depth in max_depth_list:
    for min_samples_split in min_samples_split_list:
        for seed in range(n_runs):
            accuracy.append(
                compute_DecisionTree_accuracy(x_train, y_train, x_val, y_val, max_depth, min_samples_split, seed))
        results_list.append([max_depth, min_samples_split, np.mean(accuracy)])
DecisionTrees_results_df = pd.DataFrame(results_list, columns=['max_depth', 'min_samples_split', 'accuracy'])
print (DecisionTrees_results_df)
#choosing best model within the accuracy results
#defining function for best model
def compute_best_model(results_df):
    best_model_idx = np.argmax(results_df['accuracy'])
    best_model = results_df.iloc[best_model_idx]
    return best_model
#recalling the function and passing the tree-model based accuracies to find which hyper parameters is the optimal
print('Best hyperparameter values for tree-based model: \n', compute_best_model(DecisionTrees_results_df))
#running tree-model based with values found to check the accuracy in a test set
n_runs = 5
max_depth = 25
min_samples_split = 2
test_accuracy = []
for seed in range(n_runs):
    test_accuracy.append(compute_DecisionTree_accuracy(x_train, y_train, x_test, y_test, max_depth, min_samples_split, seed))
print('Decision Tree accuracy score: \n', np.mean(test_accuracy))
#using Adaboost fro sequencial learning
#validation
#defining function to check accuracy, hyperparameters used:learning_rate, n_estimators, max_depth, min_samples_split
def compute_AdaBoost_accuracy(x_train, y_train, x_val, y_val, learning_rate, n_estimators, max_depth, min_samples_split,seed):
    base_estimator = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    model = AdaBoostClassifier(base_estimator=base_estimator, learning_rate=learning_rate, n_estimators=n_estimators, random_state=seed)
    predictions = model.fit(x_train, np.ravel(y_train)).predict(x_val)
    accuracy = accuracy_score(y_val, predictions)
    return accuracy
#checking accuracy as per tree-based model
n_runs = 3
accuracy = []
results_list = []
learning_rate_list = [0.01, 0.1, 0.5]
n_estimators_list = [10, 50, 100]
max_depth_list = [5, 10, 25]
min_samples_split_list = [2, 5, 10]
for learning_rate in learning_rate_list:
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for min_samples_split in min_samples_split_list:
                for seed in range(n_runs):
                    accuracy.append(compute_AdaBoost_accuracy(x_train, y_train, x_val, y_val, learning_rate, n_estimators, max_depth, min_samples_split, seed))
                results_list.append([learning_rate, n_estimators, max_depth, min_samples_split, np.mean(accuracy)])
AdaBoost_results_df = pd.DataFrame(results_list, columns=['learning_rate', 'n_estimators', 'max_depth', 'min_samples_split', 'accuracy'])
print(AdaBoost_results_df)
#test set
#defining function for best model
def compute_best_model(results_df):
    best_model_idx = np.argmax(results_df['accuracy'])
    best_model = results_df.iloc[best_model_idx]
    return best_model
print('Best values for Adaboost Hyperparameters: \n', compute_best_model(AdaBoost_results_df))
#checking Adaboost accuracy score passing best parameters value
n_runs = 5
learning_rate = 0.01
n_estimators = 10
max_depth = 5
min_samples_split = 10
test_accuracy = []
for seed in range(n_runs):
    test_accuracy.append(compute_AdaBoost_accuracy(x_train, y_train, x_test, y_test, learning_rate, n_estimators, max_depth, min_samples_split, seed))
print('AdaBoost accuracy score: \n', np.mean(test_accuracy))
#creating confusion matrix
base_estimator = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
model = AdaBoostClassifier(base_estimator=base_estimator, learning_rate=learning_rate, n_estimators=n_estimators, random_state=0)
fitted_model = model.fit(x_train, np.ravel(y_train))
predictions = fitted_model.predict(x_test)
print('AdaBoost Confusion Matrix: \n', confusion_matrix(y_test, predictions))
#checking features importance
feature_importances = fitted_model.feature_importances_
features_names = x_train.columns
FI = pd.DataFrame(feature_importances, index=features_names, columns=['Feature Importance']).sort_values(by='Feature Importance')
#printing dataset with features importance
print(FI)
#plotting features importances to see the most relevant for our output feature
#choose istogram graph since is more easy to see relevance of features
fig = plt.figure(figsize=(16,9))
fig.patch.set_facecolor('white')
plt.barh(range(len(features_names)), FI['Feature Importance'])
plt.yticks(range(len(features_names)), FI['Feature Importance'].index)
print(plt.title('Feature Importances'))
#using top 2 features by importance to check relation with output feature
#choose scatter comand since it better shows the relation in a 2D graph
c = ['r' if y_test.diagnosis.iloc[i] =='M' else 'g' for i in range(len(y_test))]
fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111)
fig = plt.scatter(x_test['perimeter_worst'], x_test['area_worst'],  c=c, marker=None,)
print(ax.set_xlabel('perimeter_worst'), ax.set_ylabel('area_worst'))
print(plt.show())