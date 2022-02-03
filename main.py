#importing Google dataset
import pandas as pd
df = pd.read_csv (r"D:\DataAnalysis\01_assestament\Google-Playstore.csv")

#checking shape of the dataset
print (df.shape)

# checking first 10 rows
print(df.head(10))
