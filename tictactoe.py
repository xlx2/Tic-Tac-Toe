import sys
import pygame
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector
from gymnasium import spaces


class TicTacToeEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "tictactoe",
        "fps": "120"
    }

    def __init__(self, render_mode=None):
        super().__init__()
        # 两个智能体博弈
        self.agents = ["player_1", "player_2"]  
        self.possible_agents = self.agents[:]
        # 动作空间，选择在 0～8 的位置出落子
        self.action_spaces = {
            agent: spaces.Discrete(9) for agent in self.agents
        }
        # 状态观察空间，共 9 个格，每个格子可能为 (-1, 0, 1)，
        # 0表示无子，1表示 player_1 的棋子，-1表示 player_2 的棋子
        self.observation_spaces = {
            agent: spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)
            for agent in self.agents
        }
        # 渲染相关
        self.render_mode = render_mode
        if render_mode == "human":
            self._init_pygame()
        # 重置环境
        self.reset()

    def _init_pygame(self):
        """初始化pygame界面"""
        pygame.init()
        self.WIDTH = 400
        self.HEIGHT = 400
        self.CELL_SIZE = self.WIDTH // 3
        # 设置窗口
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('Tic Tac Toe')
        # 加载图标
        try:
            icon = pygame.image.load("img/tictactoe.png")
            pygame.display.set_icon(icon)
        except:
            # 如果没有图标文件，使用默认设置
            pass
        # 棋盘图片
        self.board_img = pygame.image.load("img/board.png")
        self.board_img = pygame.transform.scale(self.board_img, (self.WIDTH, self.HEIGHT))
        # 棋子图片
        self.x_img = pygame.image.load("img/X.png")
        self.o_img = pygame.image.load("img/O.png")
        self.x_img = pygame.transform.scale(self.x_img, (self.CELL_SIZE - 20, self.CELL_SIZE - 20))
        self.o_img = pygame.transform.scale(self.o_img, (self.CELL_SIZE - 20, self.CELL_SIZE - 20))
        # 计时器
        self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        """
        初始化环境
        """
        # 初始化棋盘
        self.board = np.zeros(9, dtype=np.int8)
        # 初始化变量
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents} 
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents} 
        # 选择智能体
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # 清空pygame事件队列
        if hasattr(self, 'screen'):
            pygame.event.clear()
        
        return self.observe(self.agent_selection)

    def observe(self, agent):
        """
        返回 agent 的状态
        """
        return self.board.copy()

    def step(self, action):
        """
        多智能体环境执行一步动作
        """
        # 当前智能体
        agent = self.agent_selection
        player = 1 if agent == "player_1" else -1
        # 下在有棋子的位置是非法动作，直接结束回合
        if self.board[action] != 0:
            self._end_game(loser=agent)
            return
        # 当前智能体下棋子
        self.board[action] = player
        # 判断胜负，若有智能体胜利直接结束回合
        if self._check_win(player):
            self._end_game(winner=agent)
            return
        # 若棋盘已经下满了则平局
        if np.all(self.board != 0):
            self._draw_game()
            return
        # 切换 agent
        self.agent_selection = self._agent_selector.next()

    def _end_game(self, winner=None, loser=None):
        """
        出现赢家输家
        """
        for agent in self.agents:
            self.terminations[agent] = True
        if winner:
            self.rewards[winner] = 1
            self.rewards[self._other_agent(winner)] = -1
        elif loser:
            self.rewards[loser] = -1
            self.rewards[self._other_agent(loser)] = 1

    def _draw_game(self):
        """
        平局
        """
        for agent in self.agents:
            self.terminations[agent] = True
            self.rewards[agent] = 0

    def _other_agent(self, agent):
        """
        对方智能体
        """
        return "player_2" if agent == "player_1" else "player_1"

    def _check_win(self, player):
        """
        判断当前智能体是否胜出
        """
        board_2d = self.board.reshape(3, 3)
        for i in range(3):
            if np.all(board_2d[i, :] == player):
                return True
            if np.all(board_2d[:, i] == player):
                return True
        if np.all(np.diag(board_2d) == player):
            return True
        if np.all(np.diag(np.fliplr(board_2d)) == player):
            return True
        return False

    def render(self):
        """
        环境渲染
        """
        if self.render_mode == "human":
            self._render_pygame()
        else:
            # ansi 控制台输出版本
            symbols = {1: "X", -1: "O", 0: "."}
            b = self.board.reshape(3, 3)
            for row in b:
                print(" ".join(symbols[x] for x in row))
            print()

    def _render_pygame(self):
        """使用pygame渲染游戏界面"""
        # 检查退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # 显示棋盘背景
        if self.board_img:
            self.screen.blit(self.board_img, (0, 0))
        # 绘制棋子
        for idx in range(9):
            row, col = divmod(idx, 3)
            if self.board[idx] != 0:
                x_pos = col * self.CELL_SIZE + 10
                y_pos = row * self.CELL_SIZE + 10
                if self.board[idx] == 1:  # X
                    self.screen.blit(self.x_img, (x_pos, y_pos))
                else:  # O
                    self.screen.blit(self.o_img, (x_pos, y_pos))
        # 更新显示
        pygame.display.flip()
        self.clock.tick(120)

    def close(self):
        """关闭环境"""
        if hasattr(self, 'screen'):
            pygame.quit()