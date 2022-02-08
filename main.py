import pandas as pd
# import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
# from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

data = pd.read_csv(r"D:\DataAnalysis\01_assestament\breast-cancer.csv")
# checking tail of the dataset to calculate ideal test enviroment
print(data.tail())

