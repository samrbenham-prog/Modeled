# Modeled

Modeled is a Streamlit app that connects to Strava and estimates a runner's weekly fitness trend using training volume, pace, heart-rate efficiency, and fitness/fatigue modeling.

# Features

- Connects with your Strava
- Pulls up to 4000 Strava activities
- Converts activities into weekly training data
- Calculates training intenstiy, heart rate effiency, and a fitness/fatigue decay function
- Estimates modeled fitness score
- Shows an interactive fitness graph for each week of data
- Includes data quality checks for heart-rate coverage and number of runs

# Data Use

Modeled uses authorized Strava activity data only to calculate a personal training dashboard. It does not sell Strava data, display private activity data to other users, or use Strava data to train AI models.

# Requirements

streamlit  
pandas  
numpy  
requests  
statsmodels  
scipy  
plotly
