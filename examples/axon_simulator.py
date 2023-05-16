import matplotlib.pyplot as plt
import numpy as np
import pywt
import plotly.graph_objs as go
import plotly.express as px

from source.moreka import AxonSimulator

simulator = AxonSimulator(noise_mean=0, noise_std=0, hertz=100, duration=1000)

X, y = simulator.simulate()

plt.scatter(y.time, y.signal, alpha=0.2)

for i in range(0, len(X)):
    plt.scatter(X[i].time, X[i].signal, alpha=0.2)

plt.show()
