from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__, static_folder="templates", static_url_path="/templates")

# Load the trained models once, when the server starts
today_data = joblib.load("rain_today_model.pkl")
tomorrow_data = joblib.load("rain_tomorrow_model.pkl")


def predict_rain(temperature, humidity, pressure, wind_speed, cloud_cover):
	input_df = pd.DataFrame([{
		"Temperature": temperature,
		"Humidity": humidity,
		"AtmosphericPressure": pressure,
		"WindSpeed": wind_speed,
		"CloudCover": cloud_cover
	}])
	scaled_today = today_data["scaler"].transform(input_df)
	prob_today = today_data["model"].predict_proba(scaled_today)[0][1]
	scaled_tomorrow = tomorrow_data["scaler"].transform(input_df)
	prob_tomorrow = tomorrow_data["model"].predict_proba(scaled_tomorrow)[0][1]
	return round(prob_today * 100, 1), round(prob_tomorrow * 100, 1)


@app.route("/", methods=["GET", "POST"])
def home():
	result = None
	if request.method == "POST":
		# Read the values typed into the form
		temperature = float(request.form["temperature"])
		humidity = float(request.form["humidity"])
		pressure = float(request.form["pressure"])
		wind_speed = float(request.form["wind_speed"])
		cloud_cover = float(request.form["cloud_cover"])
		today_pct, tomorrow_pct = predict_rain(
			temperature, humidity, pressure, wind_speed, cloud_cover
		)
		result = {"today": today_pct, "tomorrow": tomorrow_pct}
	return render_template("index.html", result=result)


if __name__ == "__main__":
	app.run(debug=True)