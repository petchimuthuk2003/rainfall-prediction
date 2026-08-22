import pandas as pd

# Load data
df = pd.read_csv("data/weather.csv")

# Convert Yes/No text into 1/0 numbers
df["RainToday Num"] = df["RainToday"].map({"Yes": 1, "No": 0})
df["RainTomorrow Num"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})

# Save a cleaned version so we can reuse it later
df.to_csv("data/weather_cleaned.csv", index=False)
print("Done. Cleaned file saved as data/weather_cleaned.csv")
print(df.head())