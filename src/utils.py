import numpy as np
import matplotlib.pyplot as plt

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

def plot_results(y_test, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Temperature')
    plt.plot(predictions, label='Predicted Temperature')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()