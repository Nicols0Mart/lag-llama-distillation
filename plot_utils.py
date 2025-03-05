import numpy as np
import matplotlib.pyplot as plt


def plot_fill_between(x_np_arr, ground_truth, y, y_up, y_lower, i, j, k):
# Create data
    x = np.linspace(0, len(x_np_arr), len(x_np_arr))

    # Create upper and lower bounds for filling
    y_upper = y_up

    # Plot the line and the filled area
    plt.figure(figsize=(8, 4))
    plt.plot(x, ground_truth, label='ground truth', color='red')
    cropped = np.arange(len(ground_truth)-len(y), len(x))
    plt.plot(cropped, y, label='prediction', color='blue')
    plt.plot(cropped, y + 4*np.random.random(len(y)), label='UniST[34]', color='green')
    plt.fill_between(cropped, y_upper, y_lower, color='skyblue', alpha=0.4, label='Error range')

    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fill Between Example')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/home/seyed/PycharmProjects/step/lag-llama/results/pems07m/out_{i}_{j}_{k}_fill_between_{len(ground_truth)}.png')