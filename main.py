#importing Google dataset
import pandas as pd
df = pd.read_csv (r"D:\DataAnalysis\01_assestament\Google-Playstore.csv")

#checking shape of the dataset
print (df.shape)

#checking columns
print(df.columns)
# checking first 10 rows
print(df.head(10))
#checking missing values, only total per column
print(df.isnull().sum())

#eliminating columns that are not useful for my analysis to have a more lean database
#dropping duplicate app ID to avoid duplicate values


#filling empty places in columns with value 0 to have a complete set of data
filldata = df.fillna(0)
## print filldata to verify system has fulfilled the comand printing filldata.isnull().sum()



