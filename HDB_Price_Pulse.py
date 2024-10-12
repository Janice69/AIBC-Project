
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import logging
from utility import check_password  # Import the check_password function


# Password check at the start of the app
if not check_password():
    st.stop()

# Set the title of the Streamlit app
st.title("HDB PricePulse")

# Description
st.write(""" 
Introducing HDB PricePulse, your go-to tool for accurate HDB resale price estimates and market trends. Just input key details, and get instant price predictions based on real-time data. Plus, track and visualize market trends to stay ahead. Whether you're a buyer, seller, or investor, HDB PricePulse helps you make informed property decisions with ease. 
""")

# Define the path to your CSV file
csv_file_path = "C:\\streamlit_projects\\myenv\\Resale_Dataset.csv"

@st.cache_data
def load_data(file_path):
    """Loads data from the given CSV file and returns it as a Pandas DataFrame."""
    df = pd.read_csv(file_path)
    return df

# Fetch the data from CSV
with st.spinner("Loading data..."):
    df = load_data(csv_file_path)

# Formatting Fields
df['remaining_lease_years'] = df['remaining_lease'].str.extract(r'(\d+)').astype(float)
df['month'] = pd.to_datetime(df['month'], format='%Y-%m')
df['year'] = df['month'].dt.year
df['month_of_year'] = df['month'].dt.month
df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
df = df.dropna(subset=['resale_price', 'remaining_lease_years', 'town', 'flat_type', 'storey_range', 'floor_area_sqm'])

#################################################################################
# HDB Price Predictor #

# Train a simple linear regression model
features = ['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'year', 'month_of_year', 'remaining_lease_years']
X = df[features]
y = df['resale_price']

# Encode categorical features
X = pd.get_dummies(X, columns=['town', 'flat_type', 'storey_range'], drop_first=True)

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Title of the app

st.subheader("Resale Price Prediction Tool")
# Create input fields for features
town = st.selectbox('Town', ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                             'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
                             'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                             'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
                             'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
                             'TOA PAYOH', 'WOODLANDS', 'YISHUN'])  

flat_type = st.selectbox('Flat Type', ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 
                                       'EXECUTIVE', 'MULTI-GENERATION'])

storey_range = st.selectbox('Storey Range', ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', 
                                             '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30',
                                             '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45',
                                             '46 TO 48', '49 TO 51'])

floor_area_sqm = st.number_input('Floor Area (sqm)', min_value=0) 
remaining_lease_years = st.number_input('Remaining Lease Years', min_value=0)                                             

# Set month_of_year to the current month
month_of_year = datetime.now().month  # Automatically use the current month
# Get the current year dynamically
year = datetime.now().year  # Automatically use the current year

# Pre-processing user input for the model
# Encoding 'town'
town_dict = {
    'ANG MO KIO': 0, 'BEDOK': 1, 'BISHAN': 2, 'BUKIT BATOK': 3, 'BUKIT MERAH': 4, 'BUKIT PANJANG': 5,
    'BUKIT TIMAH': 6, 'CENTRAL AREA': 7, 'CHOA CHU KANG': 8, 'CLEMENTI': 9, 'GEYLANG': 10, 'HOUGANG': 11,
    'JURONG EAST': 12, 'JURONG WEST': 13, 'KALLANG/WHAMPOA': 14, 'MARINE PARADE': 15, 'PASIR RIS': 16,
    'PUNGGOL': 17, 'QUEENSTOWN': 18, 'SEMBAWANG': 19, 'SENGKANG': 20, 'SERANGOON': 21, 'TAMPINES': 22,
    'TOA PAYOH': 23, 'WOODLANDS': 24, 'YISHUN': 25
}
town_encoded = town_dict[town]

# Encoding 'flat_type'
flat_type_dict = {
    '1 ROOM': 0, '2 ROOM': 1, '3 ROOM': 2, '4 ROOM': 3, '5 ROOM': 4, 
    'EXECUTIVE': 5, 'MULTI-GENERATION': 6
}
flat_type_encoded = flat_type_dict[flat_type]

# Encoding 'storey_range'
storey_range_dict = {
    '01 TO 03': 0, '04 TO 06': 1, '07 TO 09': 2, '10 TO 12': 3, '13 TO 15': 4, 
    '16 TO 18': 5, '19 TO 21': 6, '22 TO 24': 7, '25 TO 27': 8, '28 TO 30': 9,
    '31 TO 33': 10, '34 TO 36': 11, '37 TO 39': 12, '40 TO 42': 13, '43 TO 45': 14, 
    '46 TO 48': 15, '49 TO 51': 16
}
storey_range_encoded = storey_range_dict[storey_range]

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'town': [town_encoded],
    'flat_type': [flat_type_encoded],
    'storey_range': [storey_range_encoded],
    'floor_area_sqm': [floor_area_sqm], 
    'year': [year],  # Automatically use the current year
    'month_of_year': [month_of_year],
    'remaining_lease_years': [remaining_lease_years]
})

# One-hot encode the input data to match training features
input_data_encoded = pd.get_dummies(input_data, columns=['town', 'flat_type', 'storey_range'], drop_first=True)

# Align the columns of input_data_encoded with those of the model
missing_cols = set(X.columns) - set(input_data_encoded.columns)
for col in missing_cols:
    input_data_encoded[col] = 0  # Add missing columns with value 0
input_data_encoded = input_data_encoded[X.columns]  # Reorder columns to match model

# Predict when the button is clicked
if st.button('Predict'):
    try:
        # Use the model to make a prediction
        prediction = model.predict(input_data_encoded)
        st.write(f'Resale Price Prediction: ${prediction[0]:,.2f}')
    except ValueError as ve:
        st.error(f"ValueError: {ve}")
    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')


##################################################################################
# Subheader for Market Trend
st.subheader("Market Trend")

# Group by year and calculate the average resale price
avg_price_by_year = df.groupby('year')['resale_price'].mean().reset_index()

# Create a line chart using Plotly for better customization
# Create the line chart using Matplotlib
# Create a Plotly figure
import plotly.graph_objects as go

# Create a Plotly figure
# Create the line chart using Matplotlib
# Create a Plotly figure
import streamlit as st
import plotly.graph_objects as go

# Create a Plotly figure
fig = go.Figure(data=[go.Scatter(x=avg_price_by_year['year'], y=avg_price_by_year['resale_price'], mode='lines+markers', marker=dict(color='blue'),
                                 text=avg_price_by_year['resale_price'],  # Add text for labels
                                 hovertemplate='Year: %{x}<br>Average Resale Price: %{y:,.0f}<extra></extra>')])  # Customize hover tooltip

# Update layout (optional)
fig.update_layout(
    title='Resale Price Trend Across Years',
    xaxis_title='Year',
    yaxis_title='Average Resale Price',
    template='plotly_dark'  # Optional: Choose a different template for styling
)

# Display the plot using st.plotly_chart
st.plotly_chart(fig)

# Flat Type Comparison:

# Filter for 2024 data
df_2024 = df[df['year'] == 2024]

# Group by flat type and calculate average resale price
avg_price_by_flat_type = df_2024.groupby('flat_type')['resale_price'].mean().reset_index()

# Create a bar chart
fig = go.Figure(data=[go.Bar(x=avg_price_by_flat_type['flat_type'], y=avg_price_by_flat_type['resale_price'],marker=dict(color='#FFC745'))])
fig.update_layout(title='Average Resale Price by Flat Type in 2024', xaxis_title='Flat Type', yaxis_title='Average Resale Price')
st.plotly_chart(fig)

#Location:
# Group by town and calculate average resale price
avg_price_by_town = df_2024.groupby('town')['resale_price'].mean().reset_index()

# Sort towns by resale price (descending)
avg_price_by_town = avg_price_by_town.sort_values(by='resale_price', ascending=False)

# Create a bar chart
fig = go.Figure(data=[go.Bar(x=avg_price_by_town['town'], y=avg_price_by_town['resale_price'],marker=dict(color='#007A78'))])
fig.update_layout(title='Average Resale Price by Town in 2024', xaxis_title='Town', yaxis_title='Average Resale Price')
st.plotly_chart(fig)
