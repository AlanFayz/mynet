import pandas as pd
import matplotlib.pyplot as plt
import time

plt.ion()  # Turn on interactive mode

fig, ax = plt.subplots()

while True:
    df = pd.read_csv("results.csv")

    ax.clear()  
    ax.plot(df['time'], df['accuracy'], marker='o', linestyle='-')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training Accuracy over Time")
    ax.grid(True)

    plt.pause(0.5)  