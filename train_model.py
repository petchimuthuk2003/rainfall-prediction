import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the cleaned data
df = pd.read_csv("data/weather_cleaned.csv")

# 2. Define our input features (same 5 columns for both models)
features = [
"Temperature", "Humidity", "AtmosphericPressure",
"WindSpeed", "CloudCover"
]
X = df[features]

#--------------------------------------------------------
# Helper function: trains one model for a given target column
#--------------------------------------------------------
def train_one_model(target_column, model_filename):
	y = df[target_column]

	# Split into training (80%) and testing (20%) sets
	# stratify=y keeps the Yes/No ratio balanced in both sets
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)

	# Scale the features so no single column (like Pressure, which has
	# big numbers like 1015) unfairly dominates the model
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	# Train a Logistic Regression model
	# class_weight="balanced" tells the model to pay extra attention to
	# the "Yes" (rain) days, since they are rarer in our data
	model = LogisticRegression(max_iter=1000, class_weight="balanced")
	model.fit(X_train_scaled, y_train)

	# Check how well it performs on data it has NOT seen before
	predictions = model.predict(X_test_scaled)
	accuracy = accuracy_score(y_test, predictions)

	print(f"\n===== Results for: {target_column} =====")
	print(f"Accuracy on unseen test data: {accuracy * 100:.1f}%")
	print(classification_report(
		y_test, predictions, target_names=["No Rain", "Rain"]
	))

	# Save the trained model AND the scaler together
	saved_data = {"model": model, "scaler": scaler, "features": features}
	joblib.dump(saved_data, model_filename)
	print(f"Saved model to: {model_filename}")

# 3. Train the "Today" model
train_one_model("RainToday Num", "rain_today_model.pkl")

# 4. Train the "Tomorrow" model
train_one_model("RainTomorrow Num", "rain_tomorrow_model.pkl")