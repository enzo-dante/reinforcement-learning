from agent import RL_TRADER

# tqdm helps w progress visualization
from tqdm import tqdm_notebook, tqdm

from utils import datasetLoader, sigmoid, stateCreator, stocksPriceFormat

# ex of apple stock dataset from yahoo datasource:
# req dataset resource from yahoo, api sends dataset res and saves to to var

# stock format = date for stock prices, high price, low price, price when market opens, price when market closes, volume of stocks to be sold, adjust close

# going to use open and close(target prediction) columns to build state for nn

stock_name = "AAPL"
data = datasetLoader(stock_name)
# print(data)

#### TRAIN RL_TRADER instance

# set hyperparams
window_size = 10
episodes = 1000  # aka epoch
batch_size = 32
data_samples = len(data) - 1

# create obj instance of RL_TRADER class
trader = RL_TRADER(window_size)

# print(trader.model.summary())

for episode in range(episodes + 1):
    print("episode: {} / {}".format(episode, episodes))

    # initial state
    state = stateCreator(data, 0, window_size + 1)

    # total profit = model progress over time
    total_profit = 0
    # start episode w 0 bought stock
    trader.inventory = []
    # timestamps = sample of timesteps(days)
    for t in tqdm(range(data_samples)):
        action = trader.trade(state)
        next_state = stateCreator(data, t + 1, window_size + 1)
        reward = 0

        # check if action performed is 1 (buy)
        if action == 1:
            trader.inventory.append(data[t])
            print("trader bought:", stocksPriceFormat(data[t]))
        elif (
            action == 2 and len(trader.inventory) > 0
        ):  # sell stock we have in inventory
            buy_price = trader.inventory.pop(0)

            reward = max(data[t] - buy_price, 0)
            total_profit += data[t] - buy_price
            print(
                "trader sold:",
                stocksPriceFormat(data[t]),
                "profit:",
                stocksPriceFormat(data[t] - buy_price),
            )
        # check if sample is last in dataset
        if t == data_samples - 1:
            done = True
        else:
            done = False

        # save obs to memory for exp replay
        trader.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("total profit: {}".format(total_profit))

        if len(trader.memory) > batch_size:
            trader.batchTrain(batch_size)
    # every 10 episodes save model
    # h5 = ext of weights
    if episode % 10 == 0:
        trader.model.save("rl_trader_{}.h5".format(episode))

