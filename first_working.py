import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from pathlib import Path
import plotly.graph_objects as go

# File path of the dataset
DATA_PATH = 'D:\\junk\\Banglore_traffic_Dataset.csv'

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    return df

def preprocess_data(df):
    os.makedirs("encoders", exist_ok=True)

    bins = [0, 30, 70, 100]
    labels = ['Low', 'Medium', 'High']
    df['CongestionCategory'] = pd.cut(df['Congestion_Level'], bins=bins, labels=labels, include_lowest=True)

    features = ['Date', 'Area_Name', 'Road_Intersection_Name', 'Weather_Conditions', 'Roadwork_and_Construction_Activity']
    X = df[features].copy()

    X['DayOfWeek'] = X['Date'].dt.dayofweek
    X['Month'] = X['Date'].dt.month
    X = X.drop(columns=['Date'])

    cat_features = ['Area_Name', 'Road_Intersection_Name', 'Weather_Conditions', 'Roadwork_and_Construction_Activity']
    for col in cat_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        joblib.dump(le, f'encoders/{col}_label_encoder.joblib')

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(df['CongestionCategory'])
    joblib.dump(label_encoder, 'encoders/congestion_label_encoder.joblib')

    return X, y_enc

def train_model(X, y):
    scaler = StandardScaler()
    clf = RandomForestClassifier(n_estimators=1000, random_state=42)

    pipeline = Pipeline([
        ('scaler', scaler),
        ('clf', clf)
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, 'rf_pipeline.joblib')

    return pipeline

def load_label_encoders():
    le_area = joblib.load('encoders/Area_Name_label_encoder.joblib')
    le_road = joblib.load('encoders/Road_Intersection_Name_label_encoder.joblib') if Path('encoders/Road_Intersection_Name_label_encoder.joblib').exists() else None
    le_weather = joblib.load('encoders/Weather_Conditions_label_encoder.joblib')
    le_roadwork = joblib.load('encoders/Roadwork_and_Construction_Activity_label_encoder.joblib')
    label_encoder = joblib.load('encoders/congestion_label_encoder.joblib')
    return le_area, le_road, le_weather, le_roadwork, label_encoder

def main():
    st.title("Traffic Congestion Prediction using Random Forest Classifier")

    df = load_data()

    # Load or train model
    try:
        pipeline = joblib.load('rf_pipeline.joblib')
        le_area, le_road, le_weather, le_roadwork, label_encoder = load_label_encoders()
    except:
        X, y = preprocess_data(df)
        pipeline = train_model(X, y)
        le_area, le_road, le_weather, le_roadwork, label_encoder = load_label_encoders()


    places = {
    "Indiranagar": ["100 Feet Road", "CMH Road"],
    "Whitefield": ["Marathahalli Bridge", "ITPL Main Road"],
    "Koramangala": ["Sony World Junction", "Sarjapur Road"],
    "M.G. Road":["Trinity Circle","Anil Kumble Circle"],
    "Jayanagar": ["Jayanagar 4th Block","South End Circle"],
    "Hebbal":["Hebbal Flyover","Ballari Road"],
    "Yeshwanthpur":["Yeshwanthpur Circle", "Tumkur Road"],
    "Electronic City": ["Hosur Road","Silk Board Junction"]
    }

    # Sidebar user inputs
    st.sidebar.header("Input traffic details:")
    date_input = st.sidebar.date_input("Date")
    area_input = st.sidebar.selectbox("Area Name", df['Area_Name'].unique())
    road_input = st.sidebar.selectbox("Road/Intersection Name", places[area_input] if area_input in places else df['Road_Intersection_Name'].unique())
    weather_input = st.sidebar.selectbox("Weather Conditions", df['Weather_Conditions'].unique())
    roadwork_input = st.sidebar.selectbox("Roadwork and Construction Activity", df['Roadwork_and_Construction_Activity'].unique())

    if not date_input or not area_input or not road_input or not weather_input or not roadwork_input:
        st.error("Please fill in all fields.")
        return

    try:
            day_of_week = pd.to_datetime(date_input).dayofweek
            month = pd.to_datetime(date_input).month

            area_encoded = le_area.transform([area_input])[0]
            road_encoded = le_road.transform([road_input])[0] if le_road else 0
            weather_encoded = le_weather.transform([weather_input])[0]
            roadwork_encoded = le_roadwork.transform([roadwork_input])[0]

            input_features = np.array([[area_encoded, road_encoded, weather_encoded, roadwork_encoded, day_of_week, month]])
            prediction_encoded = pipeline.predict(input_features)[0]

            predicted_category = label_encoder.inverse_transform([prediction_encoded])[0]

            # Map category to numeric level for gauge
            category_to_level = {'Low': 2, 'Medium': 5, 'High': 8}
            gauge_value = category_to_level.get(predicted_category, 0)

            def get_congestion_color(level):
                if level < 3:
                    return 'green'
                elif level < 6:
                    return 'yellow'
                else:
                    return 'red'

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Congestion Level"},
                gauge={
                    'axis': {'range': [0, 10], 'tickvals': [2, 5, 8], 'ticktext': ['Low', 'Medium', 'High']},
                    'bar': {'color': get_congestion_color(gauge_value)},
                    'steps': [
                        {'range': [0, 3], 'color': "lightgreen"},
                        {'range': [3, 7], 'color': "lightyellow"},
                        {'range': [7, 10], 'color': "lightcoral"},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': gauge_value
                    }
                }
            ))

            st.plotly_chart(fig)


            # Show text message for predicted category
            if predicted_category == 'Low':
                st.success('Low Traffic - Good time to travel!')
            elif predicted_category == 'Medium':
                st.warning('Moderate Traffic - Some delays expected')
            else:
                st.error('Heavy Traffic - Consider alternative routes')

            st.markdown(f"### Predicted Congestion Level: {predicted_category}")



            proba = pipeline.predict_proba(input_features)[0]
            fig = go.Figure([go.Bar(x=label_encoder.classes_, y=proba, marker_color=['red', 'green', 'yellow'])])
            fig.update_layout(title='Prediction Confidence', yaxis_title='Probability', xaxis_title='Congestion Category')
            st.plotly_chart(fig)

            


    except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

if __name__ == '__main__':
    main()
