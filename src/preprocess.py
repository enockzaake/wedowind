import h5py
import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import detrend

def extract_date(file_name):
    date_str = '_'.join(file_name.split('_')[-3:]).replace('.hdf5', '')
    return datetime.strptime(date_str, "%d_%m_%Y")

def extract_time_pd(time_str):
    hours, minutes, seconds = map(int, time_str.split('_'))
    return pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)

def calculate_rolling_mean(data, window_size):
    return data.rolling(window=window_size).mean()

def calculate_rolling_std(data, window_size):
    return data.rolling(window=window_size).std()

def check_file_exists(file_path):
    res = os.path.exists(file_path)
    if not res:
        print(f"File not found: {file_path}")
    else:
        print(f"File found: {file_path}")
    return

def process_hdf5_files(base_path, file_names, window_size=10):
    sorted_file_names = sorted(file_names, key=extract_date)
    all_features = {}

    for file_name in sorted_file_names:
        file_path = os.path.join(base_path, file_name)
        features_data = []
        file_date_str = file_name.split("Aventa_Taggenberg_")[1].split(".hdf5")[0].replace("_", "-")

        with h5py.File(file_path, "r") as f:
            print(f"Processing file: {file_path}")
            for dataset_name in f.keys():
                for time_stamp in f[dataset_name].keys():
                    signal_list = list(f[dataset_name][time_stamp].keys())

                    if 'ChannelList' in signal_list:
                        signal_list.remove('ChannelList')

                    # Check if required signals are present
                    required_signals = ['WM1', 'WM2', 'WM3', 'WM4', 'WM5', 'ATM_TEMP_01', 'GEN_ACC_XX_01']
                    if not all(signal in signal_list for signal in required_signals):
                        print(f"Skipping timestamp {time_stamp} - required signal is missing")
                        continue

                    # Extract raw data
                    wind_speed = f[dataset_name][time_stamp]['WM1']['Value'][()].flatten()
                    power_output = f[dataset_name][time_stamp]['WM2']['Value'][()].flatten()
                    rotor_speed = f[dataset_name][time_stamp]['WM3']['Value'][()].flatten()
                    temperature = f[dataset_name][time_stamp]['ATM_TEMP_01']['Value'][()].flatten()
                    rotor_acceleration = f[dataset_name][time_stamp]['GEN_ACC_XX_01']['Value'][()].flatten()

                    # Check if any of the arrays are empty
                    if len(wind_speed) == 0 or len(power_output) == 0 or len(rotor_speed) == 0 or len(temperature) == 0 or len(rotor_acceleration) == 0:
                        print(f"Skipping timestamp {time_stamp} - one or more signals are empty")
                        continue

                    # Ensure all arrays have the same length
                    min_length = min(len(wind_speed), len(power_output), len(rotor_speed), len(temperature), len(rotor_acceleration))
                    wind_speed = wind_speed[:min_length]
                    power_output = power_output[:min_length]
                    rotor_speed = rotor_speed[:min_length]
                    temperature = temperature[:min_length]
                    rotor_acceleration = rotor_acceleration[:min_length]

                    # Calculate additional features
                    wind_power_ratio = np.zeros(min_length)
                    np.divide(power_output, wind_speed, out=wind_power_ratio, where=wind_speed != 0)

                    temp_diff = np.diff(temperature)
                    temp_diff = np.append(temp_diff, 0)  # Append 0 to match the length

                    # Calculate rolling statistics
                    wind_speed_series = pd.Series(wind_speed)
                    power_output_series = pd.Series(power_output)
                    rotor_speed_series = pd.Series(rotor_speed)
                    temperature_series = pd.Series(temperature)

                    wind_speed_rolling_mean = calculate_rolling_mean(wind_speed_series, window_size)
                    wind_speed_rolling_std = calculate_rolling_std(wind_speed_series, window_size)
                    power_output_rolling_mean = calculate_rolling_mean(power_output_series, window_size)
                    power_output_rolling_std = calculate_rolling_std(power_output_series, window_size)
                    rotor_speed_rolling_mean = calculate_rolling_mean(rotor_speed_series, window_size)
                    rotor_speed_rolling_std = calculate_rolling_std(rotor_speed_series, window_size)
                    temperature_rolling_mean = calculate_rolling_mean(temperature_series, window_size)
                    temperature_rolling_std = calculate_rolling_std(temperature_series, window_size)

                    # Create a DataFrame for the features and include the formatted timestamp from the file name
                    features = pd.DataFrame({
                        'timestamp': file_date_str,
                        
                        'wind_speed': wind_speed,
                        'power_output': power_output,
                        'rotor_speed': rotor_speed,
                        'temperature': temperature,
                        'rotor_acceleration': rotor_acceleration,
                        
                        'wind_power_ratio': wind_power_ratio,
                        'temp_diff': temp_diff,
                        'wind_speed_rolling_mean': wind_speed_rolling_mean,
                        'wind_speed_rolling_std': wind_speed_rolling_std,
                        'power_output_rolling_mean': power_output_rolling_mean,
                        'power_output_rolling_std': power_output_rolling_std,
                        'rotor_speed_rolling_mean': rotor_speed_rolling_mean,
                        'rotor_speed_rolling_std': rotor_speed_rolling_std,
                        'temperature_rolling_mean': temperature_rolling_mean,
                        'temperature_rolling_std': temperature_rolling_std
                    })

                    features_data.append(features)

        # Concatenate all features DataFrames
        if features_data:
            all_features[file_name] = pd.concat(features_data, ignore_index=True)
        else:
            all_features[file_name] = pd.DataFrame()  # Empty DataFrame if no features were extracted

    return all_features

def save_features_to_csv(data, output_dir, file_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(data.values(), ignore_index=True)

    # Save the combined DataFrame to a single CSV file
    combined_csv_path = os.path.join(output_dir, f"{file_name}.csv")
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"All features combined and saved to {combined_csv_path}")

rotor_icing_file_names = [
    "Aventa_Taggenberg_03_09_2022.hdf5",
    "Aventa_Taggenberg_08_04_2022.hdf5",
    "Aventa_Taggenberg_09_04_2022.hdf5",
    "Aventa_Taggenberg_07_08_2022.hdf5",
    "Aventa_Taggenberg_01_11_2022.hdf5",
    "Aventa_Taggenberg_04_11_2022.hdf5",
    "Aventa_Taggenberg_17_12_2022.hdf5",
    "Aventa_Taggenberg_18_12_2022.hdf5",
    "Aventa_Taggenberg_19_12_2022.hdf5",
    "Aventa_Taggenberg_20_12_2022.hdf5",
]

pitch_fault_file_names = [
    "Aventa_Taggenberg_22_01_2022.hdf5",
    "Aventa_Taggenberg_23_01_2022.hdf5",
    "Aventa_Taggenberg_06_02_2022.hdf5",
    "Aventa_Taggenberg_11_02_2022.hdf5",
    "Aventa_Taggenberg_14_02_2022.hdf5",
    "Aventa_Taggenberg_15_02_2022.hdf5",
    "Aventa_Taggenberg_16_02_2022.hdf5",
    "Aventa_Taggenberg_25_02_2022.hdf5",
    "Aventa_Taggenberg_27_02_2022.hdf5",
]


imbalance_file_names = [
    "Aventa_Taggenberg_08_04_2022.hdf5",
    "Aventa_Taggenberg_09_04_2022.hdf5",
    "Aventa_Taggenberg_07_08_2022.hdf5",
    "Aventa_Taggenberg_03_09_2022.hdf5",
    "Aventa_Taggenberg_01_11_2022.hdf5",
    "Aventa_Taggenberg_04_11_2022.hdf5",
    "Aventa_Taggenberg_08_12_2022.hdf5",
    "Aventa_Taggenberg_11_12_2022.hdf5",
    "Aventa_Taggenberg_19_12_2022.hdf5",
    "Aventa_Taggenberg_23_12_2022.hdf5",
    "Aventa_Taggenberg_29_12_2022.hdf5",
    "Aventa_Taggenberg_04_01_2023.hdf5",
    "Aventa_Taggenberg_15_01_2023.hdf5",
    "Aventa_Taggenberg_21_01_2023.hdf5",
]



base_path = r"datasets/raw_data/aerodynamic_imbalance/"
output_dir = r"datasets/"

if __name__ == "__main__":
    all_features = process_hdf5_files(base_path, imbalance_file_names)
    save_features_to_csv(all_features, output_dir, "aerodynamic_imbalance")
