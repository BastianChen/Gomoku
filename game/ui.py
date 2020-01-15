import sys
import pygame
from pygame.locals import *
from game.gomoku import Game

# Window Information
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 800

MARGIN = 10
BOARD_MARGIN = 20
GRID_SIZE = WINDOW_HEIGHT - 2 * (BOARD_MARGIN + MARGIN)
PIECE_SIZE = 15
MOUSE_ACCURACY = 15

HALF_WINDOW_WIDTH = int(WINDOW_WIDTH / 2)
HALF_WINDOW_HEIGHT = int(WINDOW_HEIGHT / 2)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (180, 122, 48)
ORANGE_2 = (218, 165, 98)
ORANGE_3 = (218, 179, 130)

pygame.init()
pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

pygame.display.set_caption('Alpha Zero(五子棋)')

BASIC_FONT = pygame.font.Font('freesansbold.ttf', 16)
TITLE_FONT = pygame.font.Font('freesansbold.ttf', 24)

# 右边分数框的高度
SECTION_HEIGHT = [300, 100]
# 右边分数框的其实y坐标
SECTION_Y = [50, 400]

INFO_WIDTH = WINDOW_WIDTH - WINDOW_HEIGHT + MARGIN
INFO_X = WINDOW_HEIGHT - MARGIN


def draw(parent, child, padding=10, center=True):
    '''
    :param Parent: Rect
    :param child: Surface
    :param center: Bool
    '''
    screen = pygame.display.get_surface()
    (pw, ph), px, py = parent.size, parent.x, parent.y
    w, h = child.get_rect().size

    if center:
        x = px + (pw - w) // 2
        y = py + (ph - h) // 2
    else:
        x = px + padding
        y = py + padding

    screen.blit(child, (x, y))


class Info(pygame.sprite.Sprite):

    def __init__(self):
        super(Info, self).__init__()
        self.screen = pygame.display.get_surface()
        self.p_rect = pygame.Rect(INFO_X, 0, INFO_WIDTH, WINDOW_HEIGHT)

    def update(self, msg_list, score_record):
        self.draw_score_bg()
        self.draw_score(score_record)
        self.draw_msg(msg_list)

    def draw_score_bg(self):
        rect = pygame.Rect(INFO_X + MARGIN, SECTION_Y[0], INFO_WIDTH - 2 * MARGIN, SECTION_HEIGHT[0])
        pygame.draw.rect(self.screen, ORANGE_3, rect)

    def draw_score(self, score_record):
        w = (INFO_WIDTH - 2 * MARGIN) // 2

        h0 = PIECE_SIZE * 2
        h1 = int(SECTION_HEIGHT[0] * 0.7) - h0
        h2 = int(SECTION_HEIGHT[0] * 0.3)

        l_x = w // 2 + INFO_X + MARGIN
        r_x = w * 3 // 2 + INFO_X + MARGIN
        y = SECTION_Y[0] + MARGIN + PIECE_SIZE + 50

        pygame.draw.circle(self.screen, BLACK, (l_x, y), PIECE_SIZE, 0)
        pygame.draw.circle(self.screen, WHITE, (r_x, y), PIECE_SIZE, 0)

        l_score = BASIC_FONT.render('Human: {}'.format(score_record[0]), True, WHITE)
        r_score = BASIC_FONT.render('AlphaZero: {}'.format(score_record[1]), True, WHITE)
        b_score = BASIC_FONT.render('Tie: {}'.format(score_record[-1]), True, WHITE)

        p_rect_l = pygame.Rect(INFO_X, y + PIECE_SIZE, w, h1)
        p_rect_r = pygame.Rect(INFO_X + w + MARGIN, y + PIECE_SIZE, w, h1)
        p_rect_b = pygame.Rect(INFO_X, SECTION_Y[0] + h0 + h1, INFO_WIDTH, h2)

        draw(p_rect_l, l_score, MARGIN)
        draw(p_rect_r, r_score, MARGIN)
        draw(p_rect_b, b_score, MARGIN)

    def draw_msg(self, msg_list):
        stack_h = 0
        for msg in msg_list:
            surf = BASIC_FONT.render(msg, True, WHITE)
            stack_h += surf.get_rect().height
            p_rect = pygame.Rect(INFO_X, SECTION_Y[1] + stack_h, INFO_WIDTH, stack_h)
            draw(p_rect, surf, 0)


class BoardSprite(pygame.sprite.Sprite):

    def __init__(self, board_size, x_coord, y_coord):
        super().__init__()
        self.screen = pygame.display.get_surface()

        self.board_size = board_size
        self.x_coord = x_coord
        self.y_coord = y_coord

    def update(self, board_state):
        background = pygame.Rect(MARGIN, MARGIN, WINDOW_HEIGHT - 2 * MARGIN, WINDOW_HEIGHT - 2 * MARGIN)
        pygame.draw.rect(self.screen, ORANGE, background)

        DEV_CORRECTION = self.board_size // 3

        # Horizontal Lines
        for i in range(self.board_size):
            start_pos = (MARGIN + BOARD_MARGIN, MARGIN + BOARD_MARGIN + i * int(GRID_SIZE / (self.board_size - 1)))
            end_pos = (MARGIN + BOARD_MARGIN + GRID_SIZE - DEV_CORRECTION,
                       MARGIN + BOARD_MARGIN + i * int(GRID_SIZE / (self.board_size - 1)))
            pygame.draw.line(self.screen, BLACK, start_pos, end_pos, 1)

        # Vertical Lines
        for i in range(self.board_size):
            start_pos = (MARGIN + BOARD_MARGIN + i * int(GRID_SIZE / (self.board_size - 1)), MARGIN + BOARD_MARGIN)
            end_pos = (MARGIN + BOARD_MARGIN + i * int(GRID_SIZE / (self.board_size - 1)),
                       MARGIN + BOARD_MARGIN + GRID_SIZE - DEV_CORRECTION)
            pygame.draw.line(self.screen, BLACK, start_pos, end_pos, 1)

        # Draw pieces
        for k, _board in enumerate(board_state):
            color = BLACK if k == 0 else WHITE
            for i in range(_board.shape[0]):
                for j in range(_board.shape[1]):
                    if _board[i, j] == 1:
                        pygame.draw.circle(self.screen, color, (self.x_coord[j], self.y_coord[i]), PIECE_SIZE, 0)


class GameState:

    def __init__(self, board_size, n):
        self.screen = pygame.display.get_surface()

        self.board_size = board_size
        self.x_coord = []
        self.y_coord = []
        for i in range(board_size):
            self.x_coord.append(MARGIN + BOARD_MARGIN + i * int(GRID_SIZE / (board_size - 1)))
            self.y_coord.append(MARGIN + BOARD_MARGIN + i * int(GRID_SIZE / (board_size - 1)))

        self.info_sprite = Info()
        self.board_sprite = BoardSprite(board_size, x_coord=self.x_coord, y_coord=self.y_coord)

        self.score_record = [0, 0, 0]  # black/white/draw

        self.game = Game(board_size=board_size, n=n)

        # black: 0, white: 1
        self.start_player = 0
        self.pause = False

    @property
    def current_player(self):
        return self.game.current_player

    def next_game(self):
        self.start_player = 1 - self.start_player
        self.game.init_state()
        self.pause = False

    def step(self, action=None):
        if action is None:
            action = self.get_human_action()

        if self.pause:
            return None

        if action is not None:
            self.game.step(action)

        # Fill background color
        self.screen.fill(BLACK)

        self.board_sprite.update(self.game.state)

        terminal, winner, msg_list = self.check_win()
        if terminal:
            self.score_record[winner] += 1
            self.pause = True

        # Display Information
        self.info_sprite.update(msg_list=msg_list, score_record=self.score_record)

        pygame.display.update()

        return self.game, action

    def check_win(self):
        terminal, winner = self.game.end
        msg_list = []
        if terminal:
            if winner == 0:
                msg_list.append("Human is the winner!")
            elif winner == 1:
                msg_list.append("AlphaZero is the winner!")
            else:
                msg_list.append("Game Over, Tie!")
            msg_list.append("Press space to continue")
        else:
            player = "Black" if self.game.current_player == 0 else "White"
            msg_list.append('It\'s {} Turn'.format(player))
        return terminal, winner, msg_list

    def get_human_action(self):
        mouse_pos = 0
        for event in pygame.event.get():  # event loop
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.pause:
                    self.next_game()
            elif pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
        if mouse_pos != 0:
            for i in range(len(self.x_coord)):
                for j in range(len(self.y_coord)):
                    if ((self.x_coord[i] - MOUSE_ACCURACY < mouse_pos[0] < self.x_coord[i] + MOUSE_ACCURACY)
                            and (self.y_coord[j] - MOUSE_ACCURACY < mouse_pos[1] < self.y_coord[j] + MOUSE_ACCURACY)):

                        action = j * self.board_size + i

                        if action in self.game.action_avail:
                            return action

        return None


if __name__ == "__main__":
    gameState = GameState(15, 5)
    while True:
        gameState.step()
