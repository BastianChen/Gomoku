import numpy as np


def softmax(x):
    # return np.exp(x) / np.sum(np.exp(x))
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


# # 不加神经网络的MCTS随机生成动作以及采取该动作的概率
# def random_action(env):
#     action_probs = np.random.rand(len(env.action_avail))
#     return list(zip(env.action_avail, action_probs))
