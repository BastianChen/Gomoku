import numpy as np
import copy
from utils import *

'''
实现MCTS算法
'''


# 定义MCTS树节点
class TreeNode:
    def __init__(self, parent=None, c_puct=5.0, prob=1.0):
        # 定义父节点
        self.parent = parent
        # 定义子节点,使用字典这一数据类型
        self.children = {}
        # 该节点被访问的次数
        self.N = 0
        # 该节点的胜率（N/W，W为合计行动价值）,也叫平均行动价值
        self.Q = 0
        # 该节点的探索访问率（结合PUCT公式使用）
        self.U = 0
        # 该节点被选择的概率(用在PUCT公式中)
        self.P = prob
        # 定义探索系数
        self.c_puct = c_puct

    # 判断节点是否为叶子节点
    @property
    def is_leaf(self):
        return self.children == {}

    # # 判断节点是否为根节点
    # @property
    # def is_root(self):
    #     return self.parent is None

    # 根据PCUT公式获取价值
    # c_puct为探索系数（控制探索的强度）
    # P为节点的访问概率，访问的越多概率越大，用来控制多访问访问率高的节点
    # 最后一项分子为父节点的访问次数，分母为子节点的访问次数，随着访问次数增加，最后一项值会降低，用来提高探索访问次数少的节点
    def get_value(self):
        self.U = self.c_puct * self.P * np.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + self.U

    # 选择价值最大的子节点
    def select(self):
        # 配合key 与 lambda不等式选出PUCT值最大的动作，若只用max的话则只会根据字典的key来进行筛选，不符合我们的要求
        return max(self.children.items(), key=lambda node: node[1].get_value(self.c_puct))

    # 扩展节点，作用于叶子节点
    # action_probs包含当前状态可以选择的动作以及选择该动作的概率
    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    # 回溯树节点更新每个节点的价值
    def backup(self, value):
        # 如果存在父节点则使用递归的方式更新价值，因为是两人对弈，所以一人赢了得到的奖励为那么另一个输的人得到的奖励就为负的
        if self.parent:
            self.parent.backup(-value)
        # 此处不能用else，否则只会更新根节点的值
        self.update(value)

    # 根据最终胜负更新Q值
    def update(self, value):
        # 访问次数加一
        self.N += 1
        # 得到该节点之前的价值总和
        w = self.Q * (self.N - 1)
        # 更新该节点的Q值
        self.Q = (value + w) / self.N


# 定义MCTS来进行仿真，选择动作等操作
# policy为调用神经网络得到动作、动作概率以及评分的函数
# c_puct为探索系数
# number_playout为棋局模拟到结束的次数
class MCTS:
    def __init__(self, policy, number_playout):
        # 定义根节点
        self.root = TreeNode()
        self.policy = policy
        self.number_playout = number_playout

    # 定义棋盘模拟对弈直到结束
    def playout(self, env):
        node = self.root

        # 选择,直到叶子节点为止，根据PUCT的最大值来选择合适的动作
        while not node.is_leaf:
            action, node = node.select()
            env.step(action)

        # 通过神经网络输出动作估计概率值和当前局面价值
        # action_probs包括动作以及估计概率值
        action_probs, leaf_value = self.policy(env)

        # 判断当前棋局是否结束，如果未结束说明在叶子节点上要进行扩展，如果结束了则用胜负结果体换神经网络预测的局面价值，用于后于训练
        terminal, winner = env.end()

        if terminal:
            # 和棋
            if winner == -1:
                leaf_value = 0.
            else:
                leaf_value = 1. if winner == env.current_player else -1.
            # 回溯更新价值,因为经过step后玩家会变所以价值要加个负号
            node.backup(-leaf_value)
        else:
            node.expand(action_probs)

        # node.backup(-leaf_value)

    # 仿真，输出动作以及相应的概率
    def get_action_probs(self, env, temp=1e-3):
        # 多次仿真,每次都要深复制避免自我对弈时环境中参数变化对自我对弈产生影响
        for n in range(self.number_playout):
            env_copy = copy.deepcopy(env)
            self.playout(env_copy)

        # 根据节点被访问次数算出真实概率值
        action_visits = [(action, node.N) for action, node in self.root.children.items()]
        actions, visits = zip(*action_visits)
        action_probs = softmax(np.array(visits))
        # action_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return actions, action_probs

    # 更新MCTS
    def update_with_action(self, action):
        # 训练时从选择的动作裁剪蒙特卡洛树，节点上的值不变，为了不改变训练结果
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        # 测试时直接新建数，不让之前模拟时更新的节点数值影响下一步结果，因为模拟时随着N变大探索性变小
        else:
            self.root = TreeNode(None, 1.0)


# 定义玩家来使用MCTS
class Player:
    # is_self_play和print_detail用来区别训练和测试
    def __init__(self, policy, number_playout=2000, is_self_play=False, print_detail=False):
        self.is_self_play = is_self_play
        # 是否打印AI每个动作概率值（测试时开启）
        self.print_detail = print_detail
        # 实例化mcts算法
        self.mcts = MCTS(policy, number_playout)

    # 获取针对该玩家推荐的动作以及该动作的概率,用于被环境调用
    def get_action(self, env, temp=1e-3, return_prob=False):
        # action_avail = env.action_avail

        action_probs = np.zeros(env.width * env.height, dtype=np.float32)
        # 仿真
        actions, probs = self.mcts.get_action_probs(env, temp)
        action_probs[list(actions)] = probs
        if self.print_detail:
            self.print_probs(action_probs.reshape(env.width, env.height))

        if self.is_self_play:
            # 自我对弈训练时加入 狄利克雷分布 探索后根据概率选择动作,此时仿真已经结束，仿真时用PUCT选择动作，下子时用概率选择
            action = np.random.choice(
                actions,
                p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
            )
            self.mcts.update_with_action(action)
        else:
            # 测试时直接选择根据概率选择动作
            # action = np.random.choice(actions, p=probs)
            action = np.max(actions)
            self.mcts.update_with_action(-1)

        if return_prob:
            return action, action_probs
        else:
            return action

    # 打印出每一步的概率值
    def print_probs(self, probs):
        for i in range(probs.shape[0]):
            string = ""
            for j in range(probs.shape[1]):
                value = str(round(probs[i, j].item() * 100, 2))
                value = (" " * (6 - len(value))) + value
                string += "{} % ".format(value)
            print(string)
        print("----------------------------")
