import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from source.moreka import AxonSimulator

simulator = AxonSimulator(duration=1000, noise_std=0, noise_mean=0)

X, y = simulator.simulate()

