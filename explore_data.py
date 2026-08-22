import pandas as pd

# LoadtheCSVfile
df = pd.read_csv("data/weather.csv")

# Lookat thefirstfewrows
print(df.head())

# How manyrowsandcolumns?
print("\nShape(rows,columns):",df.shape)

# Datatypesandanymissing values
print("\nColumninfo:")
print(df.info())

# Basic statistics(average, min,max, etc.)
print("\nStatistics:")
print(df.describe())

# How manydayshadrainvsno rain?
print("\nRainTodaycounts:")
print(df["RainToday"].value_counts())

print("\nRainTomorrowcounts:")
print(df["RainTomorrow"].value_counts())