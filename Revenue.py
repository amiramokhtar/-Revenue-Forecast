import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# Load data
data = pd.read_csv("hotel_booking_data.csv")
data["date"] = pd.to_datetime(data["date"])

# Sidebar filters
st.sidebar.title("Filters")
start_date = st.sidebar.date_input("Start date", data["date"].min())
end_date = st.sidebar.date_input("End date", data["date"].max())
room_types = st.sidebar.multiselect("Room Type", data["room_type"].unique(), default=data["room_type"].unique())
countries = st.sidebar.multiselect("Country", data["country"].unique(), default=data["country"].unique())

# Filter data
filtered_data = data[
    (data["date"] >= pd.to_datetime(start_date)) &
    (data["date"] <= pd.to_datetime(end_date)) &
    (data["room_type"].isin(room_types)) &
    (data["country"].isin(countries))
]

st.title("Hotel Revenue Insights Dashboard")

# KPIs
total_bookings = filtered_data["bookings"].sum()
avg_adr = filtered_data["adr"].mean()
total_revenue = filtered_data["revenue"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Bookings", f"{total_bookings:,}")
col2.metric("Average ADR", f"${avg_adr:,.2f}")
col3.metric("Total Revenue", f"${total_revenue:,.2f}")

# Charts
st.subheader("Daily Revenue Trend")
rev_trend = filtered_data.groupby("date")["revenue"].sum().reset_index()
st.line_chart(rev_trend.set_index("date"))

st.subheader("Revenue by Room Type")
room_rev = filtered_data.groupby("room_type")["revenue"].sum().reset_index()
fig = px.bar(room_rev, x="room_type", y="revenue", color="room_type", title="Revenue by Room Type")
st.plotly_chart(fig)

st.subheader("Customer Distribution by Country")
country_dist = filtered_data["country"].value_counts().reset_index()
country_dist.columns = ["country", "count"]
fig2 = px.pie(country_dist, names="country", values="count", title="Distribution by Country")
st.plotly_chart(fig2)

# Forecasting
st.subheader("Revenue Forecast")
forecast_data = data.groupby("date")["revenue"].sum().reset_index()
forecast_data.columns = ["ds", "y"]

model = Prophet()
model.fit(forecast_data)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

fig3 = px.line(forecast, x="ds", y="yhat", title="Revenue Forecast (Next 30 Days)")
st.plotly_chart(fig3)
