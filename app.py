from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__, static_folder="templates", static_url_path="/templates")
BASE_DIR = Path(__file__).resolve().parent

# Load the trained models once, when the server starts
today_data = joblib.load(BASE_DIR / "rain_today_model.pkl")
tomorrow_data = joblib.load(BASE_DIR / "rain_tomorrow_model.pkl")


def predict_rain(temperature, humidity, pressure, wind_speed, cloud_cover):
	values = [temperature, humidity, pressure, wind_speed, cloud_cover]
	features = today_data["features"]
	input_df = pd.DataFrame([values], columns=features)
	scaled_today = today_data["scaler"].transform(input_df)
	prob_today = today_data["model"].predict_proba(scaled_today)[0][1]
	scaled_tomorrow = tomorrow_data["scaler"].transform(input_df)
	prob_tomorrow = tomorrow_data["model"].predict_proba(scaled_tomorrow)[0][1]
	return round(prob_today * 100, 1), round(prob_tomorrow * 100, 1)


@app.route("/", methods=["GET", "POST"])
def home():
	result = None
	error = None
	if request.method == "POST":
		try:
			values = [float(request.form[field]) for field in (
				"temperature", "humidity", "pressure", "wind_speed", "cloud_cover"
			)]
			today_pct, tomorrow_pct = predict_rain(*values)
			result = {"today": today_pct, "tomorrow": tomorrow_pct}
		except (KeyError, TypeError, ValueError):
			error = "Please enter a valid number in every field."
	return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
	app.run(debug=True)