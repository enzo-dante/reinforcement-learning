# math pkg access to sigmoid activation func
import math

# pdr helps w stock market info
import pandas_datareader as pdr

import numpy as np

#### dataset pre-processing
# activation func
# using sigmoid to handle price diff
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# func helps scale price to idenitfy difference between each day
def stocksPriceFormat(p):
    if p < 0:
        # if loss money, format with minus symbol and limit to 2 decimal
        return "- $ {0:2f}".format(abs(p))
    else:
        # positive numbers limited to 2 decimal points
        return "+ $ {0:2f}".format(abs(p))


# load dataset func
def datasetLoader(stock_name):
    dataset = pdr.DataReader(stock_name, data_source="yahoo")
    start_date = str(dataset.index[0]).split()[0]
    end_date = str(dataset.index[-1]).split()[0]

    close = dataset["Close"]
    return close


#### define state creator
# y-axis: prices on day
# x-axis: days
# window_size = how many previous days to include in state to predict current day price


def stateCreator(data, timestep, window_size):
    starting_id = timestep - window_size + 1

    # handle when starting_id = negative and positive
    if starting_id > 0:
        window_data = data[starting_id : timestep + 1]
    else:
        window_data = -starting_id * [data[0]] + list(data[0 : timestep + 1])

    state = []

    for i in range(window_size - 1):
        # normalize diff between next day and current day
        state.append(sigmoid(window_data[i + 1] - window_data[i]))

    return np.array([state])
