import torch
from collections import deque
from game.gomoku import Game
from net import MyNet
from mcts import Player
from config import args
from torch import nn
import os
import numpy as np
import random
from tensorboardX import SummaryWriter


class Trainer:
    def __init__(self, net_path, board_size=15, n=5):
        # 游戏棋盘大小
        self.board_size = board_size
        # 连五子胜利
        self.n = n
        # 环境实例化
        self.env = Game(board_size, n)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.number_playout = args.n_playout
        # 记忆库大小
        self.buffer_size = args.buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.batch_size = args.batch_size
        # 自我对局1次后进行训练
        self.n_games = args.n_games
        # 自我对局后进行5次训练
        self.epochs = args.epochs
        # 打印保存模型间隔
        self.check_freq = args.check_freq
        # 总共游戏次数
        self.game_num = args.game_num
        self.net_path = net_path
        self.net = MyNet().to(self.device)
        self.MSELoss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), weight_decay=1e-4)
        # 实例化蒙特卡洛玩家，参数：游戏策略，探索常数，模拟次数，是否自我对弈（测试时为False）
        self.mcts_player = Player(policy=self.policy, number_playout=self.number_playout, is_self_play=True)
        self.writer = SummaryWriter()
        if os.path.exists(net_path):
            self.net.load_state_dict(torch.load(net_path))
        else:
            self.net.apply(self.weight_init)

    def weight_init(self, net):
        if isinstance(net, nn.Linear) or isinstance(net, nn.Conv2d):
            nn.init.normal_(net.weight, mean=0., std=0.1)
            nn.init.constant_(net.bias, 0.)

    def train(self):
        for i in range(self.game_num):
            # 环境先自我对弈获得棋局状态，动作概率以及玩家可以赢的概率值
            for _ in range(self.n_games):
                winner, data = self.env.self_play(self.mcts_player, temp=1.0)
                # 打印每局对局信息
                print(self.env, "\n", "------------------xx--------")
                # 将获得的数据多样化存入样本池
                self.extend_sample(data)

            # 取样训练
            batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
            # 解包
            state_batch, mcts_probs_batch, winner_value_batch = zip(*batch)
            loss = 0.
            for _ in range(self.epochs):
                # 数据处理
                state_batch = torch.tensor(state_batch).to(self.device)
                mcts_probs_batch = torch.tensor(mcts_probs_batch).to(self.device)
                winner_value_batch = torch.tensor(winner_value_batch).to(self.device)

                # 通过神经网络输出动作概率，价值用于训练
                log_act_probs, value = self.net(state_batch)

                # 计算损失
                # 价值损失：输出价值与该状态所在对局最终胜负的值（-1/0/1）（均方差）
                # 策略损失：蒙特卡洛树模拟的概率值与神经网络模拟的概率值的相似度 (-log(pi) * p)(交叉熵)
                value_loss = self.MSELoss(value, winner_value_batch.view_as(value))
                policy_loss = -torch.mean(torch.sum(mcts_probs_batch * log_act_probs, dim=-1))
                loss = value_loss + policy_loss

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"epoch:{i},loss:{loss}")
            self.writer.add_scalar("loss", loss, i)
            self.net.add_histogram(self.writer, i)
            if (i + 1) % self.check_freq == 0:
                torch.save(self.net.state_dict(), self.net_path)

    # 多样化数据样本
    def extend_sample(self, data):
        extend_data = []
        for state, mcts_prob, winner_value in data:
            extend_data.append((state, mcts_prob, winner_value))
            # 分别旋转 90度/180度/270度，增加数据多样性
            for i in range(1, 4):
                # 同时旋转棋盘状态和概率值
                state_ = np.rot90(state, i, (1, 2))
                mcts_prob_ = np.rot90(mcts_prob.reshape(self.env.height, self.env.width), i)
                extend_data.append((state_, mcts_prob_.flatten(), winner_value))

                # 翻转棋盘,将矩阵中的每一位玩家的状态进行翻转
                state_ = np.array([np.fliplr(s) for s in state_])
                mcts_prob_ = np.fliplr(mcts_prob_)
                extend_data.append((state_, mcts_prob_.flatten(), winner_value))
        # 将样本存入样本池
        self.buffer.extend(extend_data)

    # 用于player调用神经网络获得动作概率，当前局面价值
    def policy(self, env):
        # 获取可用动作 15*15=225
        action_avail = env.action_avail
        # 获得当前状态
        state = torch.from_numpy(env.get_state).unsqueeze(0).to(self.device)

        # 放入神经网络得到预测的log动作概率以及当前状态的胜率
        log_action_probs, value = self.net(state)

        # 把 log 动作概率转换为动作概率并过滤不可用动作
        act_probs = torch.exp(log_action_probs).detach().cpu().numpy().flatten()
        act_probs = zip(action_avail, act_probs[action_avail])
        value = value.item()

        # 返回动作概率，当前局面价值
        return act_probs, value


if __name__ == '__main__':
    trainer = Trainer("models/net_7_4.pth", args.board_size, args.number)
    trainer.train()
