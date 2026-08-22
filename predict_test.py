import joblib
import pandas as pd

# Load both saved models
today_data = joblib.load("rain_today_model.pkl")
tomorrow_data = joblib.load("rain_tomorrow_model.pkl")
def predict_rain(temperature, humidity, pressure, wind_speed, cloud_cover):
	# Put the input into the same column order the model was trained on
	input_df = pd.DataFrame([{
		"Temperature": temperature,
		"Humidity": humidity,
		"AtmosphericPressure": pressure,
		"WindSpeed": wind_speed,
		"CloudCover": cloud_cover
	}])

	#----- Today----
	# predict_proba returns [prob_of_No, prob_of_Yes]; we want index [1]
	scaled_today = today_data["scaler"].transform(input_df)
	prob_today = today_data["model"].predict_proba(scaled_today)[0][1]

	#----- Tomorrow----
	scaled_tomorrow = tomorrow_data["scaler"].transform(input_df)
	prob_tomorrow = tomorrow_data["model"].predict_proba(scaled_tomorrow)[0][1]
	return round(prob_today * 100, 1), round(prob_tomorrow * 100, 1)

# Try it with an example reading
today_pct, tomorrow_pct = predict_rain(
temperature=24.5,
humidity=68,
pressure=1008.0,
wind_speed=18,
cloud_cover=6
)

print(f"Chance of rain today: {today_pct}%")
print(f"Chance of rain tomorrow: {tomorrow_pct}%")