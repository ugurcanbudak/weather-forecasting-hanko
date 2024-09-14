import pandas as pd
from datetime import datetime

# Path to your CSV file
csv_file_path = '../Weather Forecasting with LSTM/data/weather_data.csv'

# Read the CSV file
weather_df = pd.read_csv(csv_file_path)

# Process the data as needed
# Assuming the CSV has columns: "Observation station", "Year", "Month", "Day", "Time [UTC]", 
# "Average temperature [°C]", "Average relative humidity [%]", "Wind speed [m/s]", "Average air pressure [hPa]"

# Convert the "Time [UTC]" column to datetime
weather_df['datetime'] = pd.to_datetime(weather_df[['Year', 'Month', 'Day', 'Time [UTC]']].astype(str).agg(' '.join, axis=1))

# Replace non-numeric values with NaN
weather_df['Average temperature [°C]'] = pd.to_numeric(weather_df['Average temperature [°C]'], errors='coerce')
weather_df['Average relative humidity [%]'] = pd.to_numeric(weather_df['Average relative humidity [%]'], errors='coerce')
weather_df['Wind speed [m/s]'] = pd.to_numeric(weather_df['Wind speed [m/s]'], errors='coerce')
weather_df['Average air pressure [hPa]'] = pd.to_numeric(weather_df['Average air pressure [hPa]'], errors='coerce')

# Inspect the data to see how many rows have NaN values
print("Number of rows before dropping NaNs:", len(weather_df))
print("Number of NaNs in each column:")
print(weather_df.isna().sum())

# Fill NaN values with the mean of the column
weather_df['Average temperature [°C]'].fillna(weather_df['Average temperature [°C]'].mean(), inplace=True)
weather_df['Average relative humidity [%]'].fillna(weather_df['Average relative humidity [%]'].mean(), inplace=True)
weather_df['Wind speed [m/s]'].fillna(weather_df['Wind speed [m/s]'].mean(), inplace=True)
weather_df['Average air pressure [hPa]'].fillna(weather_df['Average air pressure [hPa]'].mean(), inplace=True)

# Inspect the data again to ensure NaNs are handled
print("Number of rows after filling NaNs:", len(weather_df))
print("Number of NaNs in each column after filling:")
print(weather_df.isna().sum())

# Select and reorder columns if needed
weather_df = weather_df[[
    'Observation station', 'Year', 'Month', 'Day', 'Time [UTC]', 
    'Average temperature [°C]', 'Average relative humidity [%]', 
    'Wind speed [m/s]', 'Average air pressure [hPa]'
]]

# Save the processed data to a new CSV file
output_csv_file_path = '../Weather Forecasting with LSTM/data/processed_weather_data.csv'
weather_df.to_csv(output_csv_file_path, index=False)
print(f'Processed weather data saved to {output_csv_file_path}')