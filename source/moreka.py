import os

import pandas as pd
import neo
import plotly.express as px


class AxonData:
    def __init__(self, dirname: str):
        """
        Initialize a new instance of the AxonData class.

        Args:
            dirname (str): The directory path containing the ABF files to load.
        """
        self.data = {}
        self.filenames = []
        self.load_abf_dir(dirname)

    def reset(self):
        """
        Reset the data and filenames dictionary.
        """
        self.data = {}
        self.filenames = []

    @staticmethod
    def load_abf(filename: str) -> neo.AnalogSignal:
        """
        Load an ABF file and return its AnalogSignal.

        Args:
            filename (str): The ABF file to load.

        Returns:
            A neo.AnalogSignal object.
        """
        axon_io = neo.io.AxonIO(filename=filename)
        block = axon_io.read_block(lazy=False)
        return block.segments[0].analogsignals[0]

    def load_abf_dir(self, dirname: str) -> dict:
        """
        Load all the ABF files in the specified directory.

        Args:
            dirname (str): The directory path containing the ABF files to load.

        Returns:
            A dictionary containing the loaded data.
        """
        self.filenames = [dirname + '/' + f for f in os.listdir(dirname) if f.endswith('.abf')]
        self.filenames = sorted(self.filenames)
        for filename in self.filenames:
            signal = self.load_abf(filename)
            temp_df = pd.DataFrame({'time': signal.times.reshape(-1), 'signal': signal.magnitude.reshape(-1)})
            self.data[filename] = temp_df

    def plot(self, key: int | str, **kwargs):
        """
        Plot the time and signal data for the specified file.

        Args:
            key (int | str): The index or filename of the file to plot.
            **kwargs: Optional arguments to pass to the plotly.express line chart.

        Raises:
            KeyError: If the specified key does not exist in the data dictionary.
        """
        if isinstance(key, int):
            key = self.filenames[key]
        if key not in self.data:
            raise KeyError(f"Key '{key}' not found in data dictionary.")
        df = self.data[key]

        # if kwargs contains 'downsize', then downsize the data
        if 'downsize' in kwargs:
            downsize = kwargs['downsize']
            del kwargs['downsize']
            df = df.iloc[::downsize, :]

        fig = px.line(df, x='time', y='signal', **kwargs)
        fig.show()

    def __setitem__(self, key, value):
        """
        Set the value for the specified key in the data dictionary.

        Args:
            key: The index or filename of the key to set.
            value: The value to set for the key.
        """
        if isinstance(key, int):
            self.data[self.filenames[key]] = value
        else:
            self.data[key] = value

    def __getitem__(self, key):
        """
        Get the value for the specified key from the data dictionary.

        Args:
            key: The index or filename of the key to get.

        Returns:
            The value for the specified key.
        """
        if isinstance(key, int):
            return self.data[self.filenames[key]]
        else:
            return self.data[key]

    def __repr__(self):
        """
        Return a string representation of the data dictionary.
        """
        return str(self.data)

    def __str__(self):
        """
        Return a string representation of the data dictionary.
        """
        return str(self.data)
