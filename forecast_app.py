import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st
import datetime as dt
import matplotlib.pyplot as plt
import io
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# GOOGLE SHEETS SETUP
SHEET_NAME = "BF_Forecast_Log"
WORKSHEET_NAME = "ForecastLog"

# Connect using service account
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    st.secrets["gcp_service_account"], scope)
client = gspread.authorize(creds)

# Open to Google Sheet
sheet = client.open(SHEET_NAME).worksheet(WORKSHEET_NAME)

# LOAD HISTORICAL DATA (with actuals, if available)
data = pd.DataFrame(sheet.get_all_records())

if not data.empty:
    # Convert dates
    data["Date"] = pd.to_datetime(data["Date"])

    # Rename to match (Revenue Prediction vs Actual)
    data = data.rename(columns={
        "Revenue Actual": "Revenue",
        "Tickets Actual": "Tickets"
    })

    # Convert numeric columns to floats
    numeric_cols = ["Revenue", "Tickets", "MaxTemp_F",
                    "Precipitation_Inches", "SeasonProgress"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop rows with missing actuals for model training
    train_data = data.dropna(subset=["Revenue", "Tickets"])
else:
    st.error("Google Sheet is empty. Please add historical actuals to start training.")
    st.stop()

# One-hot encoding
train_data = pd.get_dummies(
    train_data, columns=["DayOfWeek", "WeatherType"], drop_first=False)

# Drop baseline dummy variables to avoid multicollinearity
if "DayOfWeek_Monday" in train_data.columns:
    train_data = train_data.drop("DayOfWeek_Monday", axis=1)
if "WeatherType_Sunny" in train_data.columns:
    train_data = train_data.drop("WeatherType_Sunny", axis=1)

# Split features and targets
X = train_data.drop(["Revenue", "Tickets", "Date"], axis=1)
y_revenue = train_data["Revenue"]
y_tickets = train_data["Tickets"]

# Train-test split
X_train, X_test, y_revenue_train, y_revenue_test = train_test_split(
    X, y_revenue, test_size=0.2, random_state=42)
X_train2, X_test2, y_tickets_train, y_tickets_test = train_test_split(
    X, y_tickets, test_size=0.2, random_state=42)

# Train models
revenue_model = RandomForestRegressor(
    n_estimators=200, max_depth=10, random_state=42).fit(X_train, y_revenue_train)
tickets_model = RandomForestRegressor(
    n_estimators=200, max_depth=10, random_state=42).fit(X_train2, y_tickets_train)

# Evaluate
revenue_pred_test = revenue_model.predict(X_test)
tickets_pred_test = tickets_model.predict(X_test2)

st.write("**Model Accuracy Check:**")
st.write("Revenue RMSE:", np.sqrt(mean_squared_error(
    y_revenue_test, revenue_pred_test)))
st.write("Tickets RMSE:", np.sqrt(mean_squared_error(
    y_tickets_test, tickets_pred_test)))

# FETCH WEATHER FORECAST
API_KEY = st.secrets["API_KEY"]
LOCATION = "Truckee, CA"

url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{LOCATION}?unitGroup=us&key={API_KEY}&include=days&elements=datetime,tempmax,precip,preciptype"

response = requests.get(url)
forecast_json = response.json()
forecast_days = forecast_json["days"][:14]

forecast_df = pd.DataFrame(forecast_days)
forecast_df.rename(columns={
    "tempmax": "MaxTemp_F",
    "precip": "Precipitation_Inches"
}, inplace=True)

# Build features for model
forecast_df["WeatherType_Rain"] = forecast_df["preciptype"].apply(
    lambda x: 1 if x and "Rain" in str(x).lower() else 0)
forecast_df["WeatherType_Snow"] = forecast_df["preciptype"].apply(
    lambda x: 1 if x and "Snow" in str(x).lower() else 0)

forecast_df["DayOfWeek"] = pd.to_datetime(
    forecast_df["datetime"]).dt.day_name()
forecast_df = pd.get_dummies(
    forecast_df, columns=["DayOfWeek"], drop_first=False)
if "DayOfWeek_Monday" in forecast_df.columns:
    forecast_df = forecast_df.drop("DayOfWeek_Monday", axis=1)

today = dt.datetime.now().date()
season_start = dt.date(today.year, 5, 1)
forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])
forecast_df["SeasonProgress"] = (
    forecast_df["datetime"].dt.date - season_start).apply(lambda x: x.days + 1)

forecast_X = forecast_df.reindex(columns=X.columns, fill_value=0)

# Make predictions
forecast_df["Predicted Revenue"] = revenue_model.predict(forecast_X)
forecast_df["Predicted Tickets"] = tickets_model.predict(forecast_X)

# LOG PREDICTIONS TO GOOGLE SHEET
for _, row in forecast_df.iterrows():
    date_str = row["datetime"].strftime("%Y-%m-%d")

    # Check if date already exists
    existing = sheet.findall(date_str)
    if not existing:
        sheet.append_row([
            date_str,
            row["MaxTemp_F"],
            row["Precipitation_Inches"],
            "Rain" if row["WeatherType_Rain"] == 1 else (
                "Snow" if row["WeatherType_Snow"] == 1 else "Sunny"),
            row["SeasonProgress"],
            row["Predicted Revenue"],
            row["Predicted Tickets"],
            "",  # Revenue Actual (to be filled weekly)
            "",  # Ticket Actual (to be filled weekly)
            ""  # Variance filled later
        ])

# STREAMLIT DASHBOARD
st.title("14-Day Revenue & Tickets Forecast Dashboard")
st.write("Updated daily with Visual Crossing API. Model retrains on every restart.")

# Current Forecast
st.subheader("Upcoming Forecast")
st.dataframe(
    forecast_df[["datetime", "Predicted Revenue", "Predicted Tickets"]])

st.subheader("Forecast Trends")
chart_df = forecast_df[["datetime",
                        "Predicted Revenue", "Predicted Tickets"]].copy()
chart_df["datetime"] = pd.to_datetime(chart_df["datetime"])
st.line_chart(
    chart_df.set_index("datetime"),
    y=["Predicted Revenue", "Predicted Tickets"],
    use_container_width=True
)

# Historical Log (with variance)
st.subheader("Historical Predictions vs Actuals")
log_df = pd.DataFrame(sheet.get_all_records())
if not log_df.empty:
    log_df["Date"] = pd.to_datetime(log_df["Date"])
    log_df["Revenue Variance"] = log_df["Revenue Actual"].replace(
        "", np.nan).astype(float) - log_df["Revenue Prediction"].astype(float)
    log_df["Tickets Variance"] = log_df["Tickets Actual"].replace(
        "", np.nan).astype(float) - log_df["Tickets Prediction"].astype(float)
    st.dataframe(log_df)

# Feature Importance Plots


def plot_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i]
               for i in indices], rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()


st.subheader("Feature Importance")
st.write("**Revenue Model Feature Importance**")
plot_feature_importance(revenue_model, X.columns,
                        "Revenue Model Feature Importance")

st.write("**Tickets Model Feature Importance**")
plot_feature_importance(tickets_model, X.columns,
                        "Tickets Model Feature Importance")

# Download
st.subheader("Download Forecast")
csv_buffer = io.StringIO()
forecast_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="📥 Download Forecast as CSV",
    data=csv_buffer.getvalue(),
    file_name="14_day_forecast.csv",
    mime="text/csv"
)
