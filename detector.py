from mcts import Player
from game.ui import GameState
from config import args
from net import MyNet
import torch


class Detector:
    def __init__(self, net_path, board_size, n):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = MyNet().to(self.device)
        self.net.load_state_dict(torch.load(net_path))
        # self.board_size = args.board_size
        # self.n = args.number
        self.number_playout = args.n_playout
        self.env = GameState(board_size, n)
        self.net.eval()
        self.mcts_player = Player(policy=self.policy, number_playout=self.number_playout, is_self_play=True,
                                  print_detail=True)

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

    def detect(self):
        while True:
            action = None
            # 当玩家切换到人机以及游戏未结束时，人机使用MCTS算法得到最优动作
            if self.env.current_player == 1 and not self.env.pause:
                action = self.mcts_player.get_action(self.env.game)
            self.env.step(action)


if __name__ == '__main__':
    detector = Detector("models/net.pth")
    detector.detect()
