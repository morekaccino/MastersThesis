from itertools import combinations, permutations
import numpy as np
import pandas as pd
from source.moreka import AxonData
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics.pairwise import euclidean_distances
# from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
# from scipy.stats import poisson
# import tensorflow as tf
# from tensorflow import keras
# from keras.layers import Input, LSTM, RepeatVector
# from keras.models import Model
from sklearn.mixture import BayesianGaussianMixture
# import mdn
# import plotly.express as px
from sympy import symbols, Eq, solve

import warnings
import pandas as pd
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=(SettingWithCopyWarning))
# now for ConvergenceWarning
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=(ConvergenceWarning))



# from sim import generate, moving_average

for data_index in range(0, 50):
    print(f"data_index: {data_index}")
    for n_ions in range(1, 10):
        n_components = n_ions + 1
        data = AxonData(dirname='./data')
        df = data[data_index].iloc[::100, :]

        bgm = BayesianGaussianMixture(n_components=n_components, random_state=42).fit(df.signal.values.reshape(-1, 1))

        predicted = bgm.predict(df.signal.values.reshape(-1, 1))
        df['predicted'] = predicted
        conf = bgm.predict_proba(df.signal.values.reshape(-1, 1))
        # desc sort conf axis
        conf = np.sort(conf, axis=1)[:, ::-1]
        df['conf'] = conf[:, 0]

        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        domain = []
        for state_index in range(n_ions):
            p = [f"(1 - {alphabet[state_index]})", alphabet[state_index]]
            domain.append(p)

        domain = np.array(domain, dtype='str')

        equations = []

        for state_index in range(n_ions + 1):

            # print(f'# of open ions: {i}')
            equation_elements = []

            for choices_index, choices in enumerate(list(combinations(list(range(n_ions)), state_index))):
                open = domain[[choices]].reshape(state_index or n_ions, -1)[:, 1] if len(choices) else np.array([],
                                                                                                                dtype='str')
                non_choices = list(set(range(n_ions)) - set(choices))
                closed = domain[[non_choices]].reshape(n_ions - state_index, -1)[:, 0] if len(non_choices) else np.array([],
                                                                                                                         dtype='str')
                # print(choices_index, choices, open, closed)
                element = open.tolist() + closed.tolist()
                equation_elements.append(element)

            state = sorted(bgm.means_)[state_index][0]
            fake_index = np.where(bgm.means_.reshape(-1) == state)[0][0]
            equation = f"{(df[df.predicted == fake_index].shape[0]) / df.shape[0]} = " + ' + '.join(
                [' * '.join(sub_list) for sub_list in equation_elements])
            equations.append(equation)
            # print(equation)

        define_symbols_string = ", ".join(alphabet[:n_ions]) + " = symbols('" + " ".join(alphabet[:n_ions]) + "')"
        exec(define_symbols_string)

        eqs = []
        for index, equation in enumerate(equations):
            eqs.append(f"eq{index}")
            define_equation_string = f"eq{index} = Eq({equation.split('=')[1].strip()}, {equation.split('=')[0].strip()})"
            # print(define_equation_string)
            exec(define_equation_string)

        eqs = ", ".join(eqs)
        exec(f"eqs = [{eqs}]")
        vars = ", ".join(alphabet[:n_ions])
        exec(f"vars = [{vars}]")


        # print('eqs', eqs)
        # print('vars', vars)
        solutions = solve(eqs, vars, rational=False)
        if len(solutions) > 0:
            print(f"n_ions: {n_ions}")
            print(solutions)

    print('-' * 50)