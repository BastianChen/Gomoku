import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--board_size', default=15, type=int, help="棋盘大小")
parser.add_argument('--number', default=5, type=int, help="胜利条件：连续5子")
parser.add_argument('--n_playout', default=200, type=int, help="在选择动作前模拟多少次")
parser.add_argument('--c_puct', default=5, type=int, help="模拟时探索水平常数")
parser.add_argument('--buffer_size', default=20000, type=int, help="样本池大小")
parser.add_argument('--batch_size', default=512, type=int, help="批次")
parser.add_argument('--n_games', default=1, type=int, help="自我对局几次后进行训练")
parser.add_argument('--epochs', default=5, type=int, help="自我对局结束后训练几次")
parser.add_argument('--check_freq', default=10, type=int, help="打印保存模型间隔")
parser.add_argument('--game_num', default=10000, type=int, help="总共游戏次数")

args = parser.parse_args()
