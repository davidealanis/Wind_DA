import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

# Function to process the CSV file
def process_file(file):
    try:
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
    """
    Plots wind speed time series.

    Args:
      df: pandas DataFrame containing wind data.
      datetime_col: Index of the datetime column.
      speed_col: Index of the wind speed column.
    """

    # Get the name of the speed column
    speed_col_name = df.columns[speed_col]

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.iloc[:, datetime_col], df.iloc[:, speed_col], label=speed_col_name, linewidth=0.5)

    # Customize the plot
    plt.xlabel('Date')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Wind Speed Time Series')
    plt.grid(True)

    # Format the x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add legend in the upper right corner
    plt.legend(loc='upper right')

    # Display the plot in Streamlit
    st.pyplot(plt)

# Function to plot diurnal profile
def plot_diurnal_profile(df):
    # Extract datetime column dynamically
    datetime_column = next(col for col in df.columns if 'datetime' in col.lower())
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df['Hour'] = df[datetime_column].dt.hour

    # Search for wind speed column dynamically
    speed_column = next(col for col in df.columns if 'wind speed' in col.lower())

    # Group by hour and calculate mean wind speed
    hourly_mean = df.groupby('Hour')[speed_column].mean()

    # Create the plot
    plt.figure(figsize=(12, 6))
    colors = cm.Blues([0.9])

    # Plot the mean wind speed for each hour, shifting by half an hour
    x_values = hourly_mean.index + 0.5  # Half-hour shift
    plt.plot(x_values, hourly_mean, label=speed_column, color=colors[0], marker='*', linewidth=0.45)

    # Add labels and title
    plt.xlabel('Hour of Day (UTC)')
    plt.ylabel('Mean Wind Speed (m/s)')
    plt.title('Mean Diurnal Wind Speed Profile')

    # Adjust X and Y axis limits
    plt.xlim(0, 24)
    plt.xticks(range(0, 25, 1))

    # Add grid and legend
    plt.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5)
    plt.legend(loc='upper left', fontsize=8)

    # Display the plot in Streamlit
    st.pyplot(plt)

# Streamlit interface
st.title("Wind_DA - Wind data Analysis dashboard")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    st.write("File successfully uploaded. Processing data...")
    df_result = process_file(uploaded_file)

    if df_result is not None:
        st.success("File processed successfully.")
        st.write("Processed data:")
        st.dataframe(df_result)

        # Plot Time Series of wind speed (this is automatic without pressing a button)
        st.write("Wind speed time series:")
        plot_wind_speed(df_result, datetime_col=2, speed_col=1)

        # Button to plot Diurnal Profile
        if st.button("Diurnal Profile"):
            st.write("Diurnal wind speed profile:")
            plot_diurnal_profile(df_result)
