# This file was previously MainDashboard.py and is now renamed for clarity.
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import os
import google.generativeai as gen_ai
import pandas as pd  # Import Pandas for DataFrame operations


st.set_page_config(
    page_title="Vehicle Movement Analysis",
    page_icon=":car:",
    layout="wide"
)
load_dotenv()
# Retrieve Google API key from environment variables
if not os.getenv("GOOGLE_API_KEY"):
    os.environ['GOOGLE_API_KEY'] = "YOUR_GOOGLE_API_KEY"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

st.title(" Vehicle Movement Analysis")
st.markdown("""
### 📊 Vehicle Movement Patterns & Analytics
Upload your vehicle data CSV file to analyze movement patterns, authorization status, and generate insights.

**Required CSV columns:** Date, Day, Vehicle Name, Numbers, Intimes, Outtimes, Authorized, Month
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Please upload a CSV file to begin analysis.")
    st.stop()

popover = st.popover("LLM-powered insights")

# Data validation and preprocessing
try:
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    df['Intimes'] = pd.to_datetime(df['Intimes'])
    df['Outtimes'] = pd.to_datetime(df['Outtimes'])
except Exception as e:
    st.error(f"Error processing data: {e}")
    st.stop()

# Only keep page-specific controls in the sidebar
st.sidebar.title("Filters")
selected_timeframe = st.sidebar.selectbox("Select the Time frame", ['Daily','Hourly', 'Weekly', 'Yearly'])
selected_month = st.sidebar.selectbox("Select Month", df['Month'].unique())
start_date, end_date = df['Date'].min(), df['Date'].max()

filtered_df = df[(df['Month'] == selected_month) & (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

authorized_count = int(filtered_df['Authorized'].value_counts().get('Yes', 0))
unauthorized_count = int(filtered_df['Authorized'].value_counts().get('No', 0))
total_vehicles = authorized_count + unauthorized_count
date_counts = filtered_df['Date'].value_counts()
max_date = date_counts.idxmax()
min_date = date_counts.idxmin()

st.markdown("---")
st.subheader("📈 Key Performance Metrics")
metrics = st.columns([2,2,2,3])
metrics[0].metric("Total Vehicles", total_vehicles)
metrics[1].metric("Authorized Vehicles", authorized_count)
metrics[2].metric("Unauthorized Vehicles", unauthorized_count)
metrics[3].metric("Date with Most Traffic", max_date.strftime("%Y-%m-%d"), int(date_counts.max()))

st.markdown("---")
st.subheader("🚗 Vehicle Entries Over Time")
if selected_timeframe == 'Hourly':
    df_resampled = filtered_df.set_index('Intimes').resample('H').size().to_frame(name='count').reset_index()
    x_axis = 'Intimes'
elif selected_timeframe == 'Daily':
    df_resampled = filtered_df.set_index('Date').resample('D').size().to_frame(name='count').reset_index()
    x_axis = 'Date'
elif selected_timeframe == 'Weekly':
    df_resampled = filtered_df.set_index('Date').resample('W-MON').size().to_frame(name='count').reset_index()
    x_axis = 'Date'
elif selected_timeframe == 'Yearly':
    df_resampled = filtered_df.set_index('Date').resample('Y').size().to_frame(name='count').reset_index()
    x_axis = 'Date'

fig = px.line(df_resampled, x=x_axis, y='count', title=f'Vehicle Entries Over Time ({selected_timeframe})')
st.plotly_chart(fig)

st.markdown("---")
st.subheader("🚦 Entry & Exit Traffic Patterns")
fig = make_subplots(rows=1, cols=2, subplot_titles=("Entry Hour (6am - 6pm)", "Exit Hour (6am - 6pm)"))

entry_hour = filtered_df['Intimes'].dt.hour
exit_hour = filtered_df['Outtimes'].dt.hour

fig.add_trace(go.Histogram(x=entry_hour, nbinsx=12, name='Entry Hour'), row=1, col=1)
fig.add_trace(go.Histogram(x=exit_hour, nbinsx=12, name='Exit Hour'), row=1, col=2)

fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.update_layout(height=500, width=900, title_text="Frequency of Vehicles by Entry and Exit Hours")
st.plotly_chart(fig)

st.markdown("---")
st.subheader("✅ Authorization Status & Vehicle Types")
fig1, fig2 = st.columns(2)
with fig1:
    auth_counts = filtered_df['Authorized'].value_counts()
    fig = px.pie(values=auth_counts, names=auth_counts.index, title='Authorized/Not Authorized')
    st.plotly_chart(fig)

with fig2:
    vehicle_type_counts = filtered_df['Vehicle Name'].value_counts()
    fig = px.pie(values=vehicle_type_counts, names=vehicle_type_counts.index, title='Vehicle Types')
    st.plotly_chart(fig)

st.markdown("---")
st.subheader("📅 Weekly Vehicle Frequency Analysis")
filtered_df = df[df['Month'] == selected_month]

filtered_df = filtered_df[(filtered_df['Date'] >= pd.to_datetime(start_date)) & (filtered_df['Date'] <= pd.to_datetime(end_date))]
day_counts = filtered_df['Day'].value_counts()
fig = px.bar(day_counts, x=day_counts.index, y=day_counts.values, 
            labels={'x':'Day', 'y':'Frequency'}, title=f'Frequency of Vehicles by Day of the Week ({selected_month})')
st.plotly_chart(fig)

# Set up Google Gemini-Pro AI model
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')

def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

with popover:
    st.title("🤖 AI-Powered Insights Chat")
    st.markdown("Ask questions about your vehicle movement data using natural language.")

    prompt = f"Here is a CSV file containing data about intimes, outtimes, vehicle types and frequencies of vehicles grouped by days of the month in the YYYY-MM-DD format, with columns {df.columns} and data {df}.As an expert data analyst, Please analyze it and provide insights only in regards with the following question:"

    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text.lstrip(prompt))
    user_prompt = st.chat_input("💬 Ask questions about vehicle patterns, trends, or insights...")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        gemini_response = st.session_state.chat_session.send_message(prompt+user_prompt)
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)
