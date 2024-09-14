# Weather Forecasting of Hanko, Finland with LSTM

This project uses Long Short-Term Memory (LSTM) neural networks to forecast weather temperatures of Hanko, Finland based on historical data.

## Description

The project involves:
- Loading and preprocessing weather data.
- Normalizing the data.
- Preparing sequences for LSTM input.
- Building and tuning an LSTM model using Keras Tuner.
- Training and evaluating the model.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/weather-forecasting-lstm.git
    cd weather-forecasting-lstm
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Ensure you have the processed weather data file at `data/processed_weather_data.csv`.

2. Run the training script:
    ```sh
    python train.py
    ```

3. The script will load the data, preprocess it, build and tune the LSTM model, and train it.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License.
