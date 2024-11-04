from flask import Flask, render_template, request 
import plotly.express as px
from plotly.io import to_html
from plotly.subplots import make_subplots
import pandas as pd
import json
import plotly.graph_objects as go
from joblib import load
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer, StandardScaler, OneHotEncoder, LabelEncoder
app = Flask(__name__)

# File paths
geojson_path = 'data/map.geojson'
model_path = 'model.joblib'

# Load traffic data and preprocess time column
data_frame = pd.read_csv('data/Traffic.csv') 

# Load GeoJSON data
def load_geojson():
    with open(geojson_path) as f:
        return json.load(f)

# Load trained model
def load_model():
    return load(model_path)

scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1)
# Fetch traffic data
def preprocess_data(data, hour, minute, day):
    filtered_data = data[(
            pd.to_datetime(data['Time'], format='%I:%M:%S %p').dt.hour == hour) & 
            (pd.to_datetime(data['Time'], format='%I:%M:%S %p').dt.minute == minute) & 
            (data['Date'] == day)
        ].copy()
    filtered_data['Weekend'] = filtered_data['Day of the week'].isin(['Saturday', 'Sunday'])
    filtered_data['Hour'] = pd.to_datetime(filtered_data['Time'], format='%I:%M:%S %p').dt.hour
    filtered_data['Traffic Situation'] = filtered_data['Traffic Situation'].astype('category').cat.codes
    filtered_data[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']] = scaler.fit_transform(filtered_data[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']])
    filtered_data = filtered_data.drop(columns=['Traffic Situation'])   
    

    return filtered_data


@app.route('/', methods=['GET', 'POST'])
def index():
    # Load geojson and model
    geojson_data = load_geojson()
    model = load_model()

    # Simulate choropleth data for the map
    data = {
        'name': ['Heavy', 'Low', 'Normal', 'High'],
        'value': [0,1,2,3]
    }
    df = pd.DataFrame(data)

    # Plot the choropleth map
    fig = px.choropleth_mapbox(
        df,
        geojson=geojson_data,
        locations='name',
        color='name',
        mapbox_style='open-street-map',
        zoom=15,
        center={"lat": 10.7835, "lon": 106.69650},
        opacity=0.5,
        labels={'name': 'Traffic Situation'},
        width=800,
        height=500, 
        color_discrete_map={
        "Heavy": "red",    
        "Low": "green",       
        "Normal": "blue",      
        'High': 'yellow'
        },
    )

    # Add a line trace for Hai Bà Trưng street
    line_data = pd.DataFrame({
        'lat': [10.7850, 10.7816],
        'lon': [106.6956, 106.6995],
        'name': ['Start', 'End']
    })
    fig.add_trace(go.Scattermapbox(
        lat=line_data['lat'],
        lon=line_data['lon'],
        mode='lines+markers',
        marker=go.scattermapbox.Marker(size=8, color='red'),
        line=dict(width=4, color='blue'),
        name='Hai Bà Trưng'
    ))

    # Generate traffic situation pie chart
    fig_pie = px.pie(data_frame, names='Traffic Situation', title='Traffic Situation Distribution', color_discrete_sequence=px.colors.sequential.RdBu)
    fig_pie.update_layout(title_text='Traffic Situation Distribution', title_x=0.5, template='plotly_white')
    fig_pie_html = to_html(fig_pie, full_html=False)

    # 5-8. Relationship between vehicle counts and traffic situation
    fig_relationship = make_subplots(rows=2, cols=2, subplot_titles=("Car", "Bike", "Bus", "Truck"))

    fig_relationship.add_trace(go.Box(x=data_frame['Traffic Situation'], y=data_frame['CarCount'], name='Car Counts', marker_color='#1f77b4'), row=1, col=1)
    fig_relationship.add_trace(go.Box(x=data_frame['Traffic Situation'], y=data_frame['BikeCount'], name='Bike Counts', marker_color='#ff7f0e'), row=1, col=2)
    fig_relationship.add_trace(go.Box(x=data_frame['Traffic Situation'], y=data_frame['BusCount'], name='Bus Counts', marker_color='#2ca02c'), row=2, col=1)
    fig_relationship.add_trace(go.Box(x=data_frame['Traffic Situation'], y=data_frame['TruckCount'], name='Truck Counts', marker_color='#d62728'), row=2, col=2)

    fig_relationship.update_layout(title_text='Vehicle Counts by Traffic Situation', title_x=0.5, showlegend=False, template='plotly_white')
    fig_relationship.update_xaxes(title_text="Traffic Situation")
    fig_relationship.update_yaxes(title_text="Count")
    fig_relationship_html = to_html(fig_relationship, full_html=False)

    # 11. Busiest hours of the day for traffic
    data_frame['Hour'] = pd.to_datetime(data_frame['Time'], format='%I:%M:%S %p').dt.hour
    fig_hour = px.box(data_frame, x='Hour', y='Total', title='Total Vehicle Count by Hour', color_discrete_sequence=['#9467bd'])
    fig_hour.update_layout(title_text='Total Vehicle Count by Hour', title_x=0.5, xaxis_title='Hour', yaxis_title='Total Count', template='plotly_white',width=915, height=320)
    fig_hour_html = to_html(fig_hour, full_html=False)

    # graph_html = fig.to_html(full_html=False)
    # Convert figure to HTML
    graph_html = fig.to_html(full_html=False)

    filtered_data_html = ""
    predicted_output = ""
    output = ''

    if request.method == 'POST':
        # Get user input from form
        day = int(request.form['day'])
        hour = int(request.form['hour'])
        am_pm = request.form['am_pm']
        minute = int(request.form['minute'])

        # Convert to 24-hour format
        if am_pm == 'PM' and hour < 12:
            hour += 12
        if am_pm == 'AM' and hour == 12:
            hour = 0

        input_features = preprocess_data(data_frame, hour=hour, minute=minute, day=day)

        # Filter the data based on user input
        

        # Check if data is available
        if not input_features.empty:

            input = input_features.iloc[0].to_dict()
            input_features_df = pd.DataFrame([input])

            predicted_output = model.predict(input_features_df)[0]
            # print(predicted_output)
            inverse_transformed_data = scaler.inverse_transform(input_features[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']])
            input_features[['CarCount', 'BikeCount', 'BusCount', 'TruckCount']] = inverse_transformed_data


            # Define color map for prediction output
            color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow'}
            output_map = {0: 'Heavy', 1: 'Low', 2: 'Normal', 3: 'High'}
            line_color = color_map.get(predicted_output, 'gray')
            output = output_map.get(predicted_output, 'None')

            # Update line color in the map for prediction
            fig.data[-1].line.color = line_color

            # Convert filtered data to HTML table
            filtered_data_html = input_features.iloc[:, :-3].to_html(classes='table table-striped', index=False)
        else:
            predicted_output = "Không có dữ liệu cho thời gian và ngày được chọn."
            line_color = 'gray'
            output = 'None'

    # Update the map with the new trace
    graph_html = fig.to_html(full_html=False)

    # Render the template with the prediction result
    return render_template('index.html', graph_html=graph_html, filtered_data_html=filtered_data_html, output=output, fig_pie_html=fig_pie_html, fig_relationship_html=fig_relationship_html, fig_hour_html=fig_hour_html)

if __name__ == '__main__':
    app.run(debug=True)
