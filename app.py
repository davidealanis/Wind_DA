import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import requests
from io import StringIO  # Para manejar el contenido de archivos como cadenas de texto
from scipy.stats import weibull_min  # Para ajustar la distribución de Weibull
import numpy as np

# Function to process the CSV file
def process_file(file):
    try:
        # Check if file is a StringIO object (from GitHub) or an uploaded file
        if isinstance(file, StringIO):
            # If it's a StringIO object (from GitHub), read the content as a string
            lines = file.getvalue().splitlines()
        else:
            # Otherwise, decode the uploaded file as usual
            lines = file.read().decode("utf-8").splitlines()

        # Look for the header row
        header_row = None
        for idx, line in enumerate(lines):
            if "wind direction" in line.lower() and "wind speed" in line.lower():
                header_row = idx
                break
        
        if header_row is None:
            st.error("No header related to wind was found.")
            return None

        file.seek(0)
        df = pd.read_csv(file, header=header_row)

        columns_of_interest = [col for col in df.columns if 'wind' in col.lower()]
        if not columns_of_interest:
            st.error("No columns related to wind were found.")
            return None
        
        df_wind = df[columns_of_interest].copy()

        for col in ['Year', 'Month', 'Day', 'Hour', 'Minute']:
            if col in df.columns:
                df_wind[col] = df[col]
            else:
                st.error(f"The column {col} is not present in the data.")
                return None

        df_wind['datetime'] = pd.to_datetime(df_wind[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        df_wind = df_wind.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])

        return df_wind
    
    except pd.errors.ParserError as e:
        st.error(f"Error processing the file: {e}")
        return None

# Function to generate the wind speed time series plot
def plot_wind_speed(df, datetime_col, speed_col):
    speed_col_name = df.columns[speed_col]

    plt.figure(figsize=(12, 6))
    plt.plot(df.iloc[:, datetime_col], df.iloc[:, speed_col], label=speed_col_name, linewidth=0.5)

    plt.xlabel('Date')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Wind Speed Time Series')
    plt.grid(True)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)

    plt.legend(loc='upper right')
    st.pyplot(plt)

# Function to plot diurnal profile
def plot_diurnal_profile(df):
    datetime_column = next(col for col in df.columns if 'datetime' in col.lower())
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df['Hour'] = df[datetime_column].dt.hour

    speed_column = next(col for col in df.columns if 'wind speed' in col.lower())

    hourly_mean = df.groupby('Hour')[speed_column].mean()

    plt.figure(figsize=(12, 6))
    colors = cm.Blues([0.9])

    x_values = hourly_mean.index + 0.5  # Half-hour shift
    plt.plot(x_values, hourly_mean, label=speed_column, color=colors[0], marker='*', linewidth=0.45)

    plt.xlabel('Hour of Day (UTC)')
    plt.ylabel('Mean Wind Speed (m/s)')
    plt.title('Mean Diurnal Wind Speed Profile')
    plt.xlim(0, 24)
    plt.xticks(range(0, 25, 1))

    plt.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper left', fontsize=8)

    st.pyplot(plt)

# Function to calculate and display Weibull distribution parameters
def calculate_weibull(df):
    speed_column = next(col for col in df.columns if 'wind speed' in col.lower())
    wind_speeds = df[speed_column].dropna()

    # Fit Weibull distribution
    params = weibull_min.fit(wind_speeds, floc=0)  # floc=0 forces location parameter to be 0
    shape, loc, scale = params

    # Create a summary table
    weibull_summary = pd.DataFrame({
        'Parameter': ['Shape (k)', 'Scale (c)'],
        'Value': [shape, scale]
    })

    st.write("### Weibull Distribution Parameters")
    st.table(weibull_summary)

    # Plot the histogram and Weibull
    #plt.figure(figsize=(10, 6))
    #plt.hist(wind_speeds, bins=30, density=True, alpha=0.6, color='lightblue',edgecolor = 'black', label='Wind Speed Histogram')

    # Plot the histogram and Weibull
    plt.figure(figsize=(10, 6))
    # Calcular histograma en porcentaje
    hist_values, bin_edges = np.histogram(wind_speeds, bins=30, density=True)
    hist_values = hist_values * 100  # Convertimos a porcentaje
    # Graficamos el histograma corregido
    plt.bar(bin_edges[:-1], hist_values, width=np.diff(bin_edges), color='lightblue', edgecolor='black', alpha=0.6, label='Wind Speed Histogram', align='edge')

    
    #Weibull curve
    x = np.linspace(wind_speeds.min(), wind_speeds.max(), 100)
    pdf = weibull_min.pdf(x, shape, loc, scale)*100 #Probability density funciton
    plt.plot(x, pdf, color='black', label='Weibull')

    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Density (%)')
    plt.title('Weibull Distribution Fit')
    plt.legend()
    st.pyplot(plt)

# Streamlit interface
st.title("Wind_DA")
st.title("Wind data analysis dashboard")
# Dashboard description
st.markdown("""
**Wind_DA** is an interactive tool for analyzing wind data, 
including time series and diurnal profiles. 
Developed by **David Alanis**.
""")

st.markdown("""
### About this tool:
- **Purpose**: Analyze wind data to understand patterns in speed and direction.
- **Features**:
    - Visualization of wind speed time series.
    - Generation of diurnal wind speed profiles.
    - Calculation of Weibull distribution parameters.
- **Developer**: David Alanis
- **Source code**: [GitHub](https://github.com/davidealanis/Wind_DA)
""")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Option to load the sample file from GitHub
use_sample_file = st.checkbox("Use sample file from GitHub")

if use_sample_file:
    url = "https://raw.githubusercontent.com/davidealanis/Wind_DA/main/1745739_35.29_-87.68_2014.csv"  # Replace with your GitHub file URL
    response = requests.get(url)
    uploaded_file = StringIO(response.text)  # Convert response to StringIO for CSV processing

if uploaded_file:
    st.write("File successfully uploaded. Processing data...")
    df_result = process_file(uploaded_file)

    if df_result is not None:
        st.success("File processed successfully.")
        st.write("Processed data:")
        st.dataframe(df_result)

        # Plot Time Series of wind speed (this is automatic without pressing a button)
        st.write("### Wind speed time series:")
        plot_wind_speed(df_result, datetime_col=2, speed_col=1)

        # Button to plot Diurnal Profile
        #'''if st.button("Diurnal Profile"):
        #    st.write("Diurnal wind speed profile:")
        #    plot_diurnal_profile(df_result)'''
        
        st.write("### Diurnal wind speed profile:")
        plot_diurnal_profile(df_result)

        # Calculate and display Weibull parameters
        #if st.button("Weibull Parameters"):
        #    calculate_weibull(df_result)'''
        #st.button("Weibull Parameters"):
        calculate_weibull(df_result)