from flask import Flask, render_template
import pandas as pd
import plotly.express as px
from prophet import Prophet

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('sales.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Prepare the data for Prophet
df_prophet = df.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})

# Initialize and fit the model
model = Prophet()
model.fit(df_prophet)

@app.route('/')
def index():
    # Create future dates and forecast
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    
    # Create plotly figure
    fig = px.line(forecast, x='ds', y='yhat', title='Demand Forecast')
    graph = fig.to_html(full_html=False)
    
    return render_template('index.html', graph=graph)

if __name__ == '__main__':
    app.run(debug=True)
