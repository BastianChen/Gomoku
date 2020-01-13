import numpy as np
import torch

'''
定义环境（此处为棋盘），需要有判断落子在什么位置的函数，是否结束游戏，返回环境状态等函数
定义调用环境的Game类，需要有使用环境的类，根据环境返回的状态给出回报值等函数
'''


class Game:
    def __init__(self, board_size=6, n=4, start_player=0):
        self.width, self.height = board_size, board_size
        # 定义连续n子为胜利条件
        self.n = n
        # 定义棋盘大小
        self.size = (self.width, self.height)
        # 定义两个玩家
        self.players = [0, 1]
        # 定义当前玩家
        self.current_player = start_player
        # 定义可选动作
        self.action_avail = list(range(self.size[0] * self.size[1]))
        # 定义state，大小为[n,2,9,9]，两个通道表示两个玩家，下过子的地方值为1，没下过的地方值为0
        self.state = np.zeros([2, *self.size], dtype=np.float32)
        # 定义上一步动作
        self.last_action = -1

    # 初始化参数用于自训练
    def init_state(self):
        # 可选动作
        self.action_avail = list(range(self.size[0] * self.size[1]))
        # state大小为n[n,2,9,9]
        self.state = np.zeros((2, *self.size), dtype=np.float32)
        # self.last_action = -1

    # 根据不同的玩家返回不同的state
    @property
    def get_state(self):
        if self.current_player == 0:
            state = self.state.copy().astype(np.float32)
        else:
            # 沿着指定的轴反转元素,将玩家1的状态放在上面，来让神经网络区别同一棋盘状态下不同玩家的状态
            state = np.flip(self.state, axis=0).astype(np.float32)
        return state

    # 根据动作数在棋盘上找到相应的位置
    def action_to_location(self, action):
        x = action % self.size[1]
        y = action // self.size[0]
        return [y, x]

    # 采取动作
    def step(self, action):
        # 将动作对应到棋盘上的具体位置
        location = self.action_to_location(action)
        # 在对应玩家状态的对应位置上置1，表示该点下过子
        self.state[self.current_player][location[0]][location[1]] = 1.
        # 在可选择的动作中删除已经使用过的动作
        self.action_avail.remove(action)
        self.last_action = action
        # 切换另一名玩家下棋
        self.current_player = 1 - self.current_player
        # 返回是否结束的结果以及胜利的玩家
        return self.end

    # 生成可以胜利的落子条件
    def generate(self, action):
        result = []
        x = action % self.size[1]
        y = action // self.size[0]

        # 横向连续
        if x <= self.width - self.n:
            result.append([index for index in range(action, action + self.n)])
        # 纵向连续
        if y <= self.height - self.n:
            result.append([index for index in range(action, action + self.n * self.width, self.width)])
        # 右对角线连续,n-1是因为棋盘的索引是从0开始的
        if x >= self.n - 1 and y <= self.height - self.n:
            result.append([index for index in range(action, action + self.n * (self.width - 1), self.width - 1)])
        # 左对角线连续
        if x <= self.width - self.n and y <= self.height - self.n:
            result.append([index for index in range(action, action + self.n * (self.width + 1), self.width + 1)])

        return np.array(result)

    # 判断是否胜利,根据多个连子情况乘积的总和是否大于0来判断
    @property
    def check_win(self):
        # 遍历每个玩家的状态
        for player in self.players:
            state_ = self.state[player]
            # 获取状态中非零处的索引，即落了子的地方
            actions = state_.flatten().nonzero()[0]
            # 遍历每个落了子的索引，再在状态选取可以结束游戏的连子条件，判断多个连子情况乘积的总和是否大于0
            for action in actions:
                indexs = self.generate(action)
                if len(indexs) == 0:
                    continue
                else:
                    value = state_.take(indexs)
                    # 因为可能返回多个连子情况，所以可以根据多个连子情况乘积的总和是否大于0来判断是否结束
                    value = value.prod(axis=-1)
                    value = value.sum()
                if value > 0:
                    return True

    # 判断是否结束,并返回是否结束以及赢的玩家
    @property
    def end(self):
        if self.check_win:
            # 因为在落完子后会切换用户，但是结束子是没转换前的用户下的，所以要切换回用户
            return True, 1 - self.current_player
        # 没有地方可以下子了，表示游戏结束但是为和棋
        elif len(self.action_avail) == 0:
            return True, -1
        else:
            return False, -1

    # 使用根据MCTS获得的动作进行自我对弈（在线学习）
    # 参数：player为玩家，temp为概率修正系数，is_shown为是否打印棋盘信息
    # 一个玩家对应一个MCTS
    def self_play(self, player, temp=1e-3, is_shown=True):
        # 初始化棋盘信息
        self.init_state()
        # 创建list保存在自训练过程中产生的状态，mcts对该动作预测的概率以及对应的玩家
        states, mcts_probs, current_players = [], [], []
        # 一直自我对弈直到游戏结束
        while True:
            # 获取针对该玩家推荐的动作以及该动作的概率
            action, action_probs = player.get_action(self, temp=temp, return_prob=True)
            states.append(self.get_state)
            mcts_probs.append(action_probs)
            current_players.append(self.current_player)

            # 根据推荐的动作执行下一步
            terminal, winner = self.step(action)
            # 判断是否要打印棋盘信息，测试的时候打印，训练的时候不打印
            if is_shown:
                print(self)

            # 判断是否结束,并返回赢家，当前棋盘状态，选择该动作的概率以及分数
            if terminal:
                winner_value = np.zeros(len(current_players))
                # 判断是否和棋，若为和棋则两个玩家都得0分
                if winner != -1:
                    winner_value[np.array(current_players) == winner] = 1.0
                    winner_value[np.array(current_players) != winner] = -1.0

                # 重置用户的蒙特卡洛搜索树
                player.reset_player()
                # 打印游戏结果
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                # 返回结果用于后面神经网络采样训练
                return winner, list(zip(states, mcts_probs, winner_value))

    # 重写str函数
    def __str__(self):
        string = ''
        for i in range(self.height):
            row = ''
            for j in range(self.width):
                # 遍历出第一个玩家下的子
                if self.state[0, i, j] == 1:
                    row += "o"
                # 遍历出第二个玩家下的子
                elif self.state[1, i, j] == 1:
                    row += "x"
                # 还未下子的地方
                else:
                    row += "-"
                row += " "
            row += "\n"
            string += row
        return string


if __name__ == '__main__':
    game = Game()
    game.step(8)
    game.step(9)
    game.step(15)
    game.step(14)
    game.step(20)
    game.step(21)
    game.step(10)
    game.step(5)
    game.step(25)
    print(game)
    print(game.end)
