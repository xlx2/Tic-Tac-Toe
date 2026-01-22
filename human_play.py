import pygame
import numpy as np
from tictactoe import TicTacToeEnv
from utils import get_valid_actions, load_model


class HumanPlayer:
    """人类玩家类"""
    def __init__(self, env, player_id):
        self.env = env
        self.player_id = player_id
        self.is_human = True

    def get_action(self, state):
        """获取人类玩家的行动"""
        # 将一维状态转换为二维便于处理
        board_2d = np.array(state).reshape(3, 3)
        
        # 检查pygame事件以获取鼠标点击
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # 鼠标左键点击
                x, y = event.pos
                col = x // (self.env.WIDTH // 3)
                row = y // (self.env.HEIGHT // 3)
                # 将2D坐标转换为1D动作
                action = row * 3 + col
                # 检查动作是否有效
                if board_2d[row][col] == 0:  # 如果位置为空
                    return action
        
        return None  # 没有有效动作


def human_vs_ai_game():
    """人类与AI对战的主循环"""
    print("\n欢迎与AI对战！")
    print("点击棋盘区域来下棋")
    # 加载AI模型
    try:
        q_tables = load_model("final_q_tables.pkl")
        print("AI模型加载成功！")
    except FileNotFoundError:
        print("模型文件不存在，使用随机策略。")
        q_tables = None
    # 创建带渲染模式的环境
    env = TicTacToeEnv(render_mode="human")
    obs = env.reset()
    # 人类玩家选择符号
    human_choice = input("请选择先手 (输入 '1') 或后手 (输入 '2'), 默认为先手: ") or "1"
    if human_choice == '1':
        human_agent = "player_1"
        ai_agent = "player_2"
        print("你是 X")
    else:
        human_agent = "player_2"
        ai_agent = "player_1"
        print("你是 O")
    # 创建人类玩家
    human_player = HumanPlayer(env, human_agent)
    game_over = False
    # 主游戏循环
    while not game_over:
        # 渲染当前状态
        env.render()
        # 获取当前代理
        current_agent = env.agent_selection
        if current_agent == human_agent:
            # 人类玩家回合
            action = None
            while action is None:
                action = human_player.get_action(obs)
                if action is not None:
                    break
        else:
            # AI回合
            if q_tables is not None:
                # 使用训练好的AI策略
                state_tuple = tuple(obs)
                valid_actions = get_valid_actions((state_tuple, None))
                if len(valid_actions) > 0:
                    # 获取Q值最高的动作
                    q_values = [q_tables[ai_agent].get((state_tuple, a), 0) for a in valid_actions]
                    max_q = max(q_values) if q_values else 0
                    best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
                    action = best_actions[0]  # 选择第一个最佳动作
                else:
                    print("没有可用的动作")
                    break
            else:
                # 使用随机策略
                valid_actions = [i for i in range(9) if obs[i] == 0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    print("没有可用的动作")
                    break
        # 执行动作
        env.step(action)
        obs = env.observe(current_agent)
        
        # 检查游戏是否结束
        if any(env.terminations.values()):
            env.render()
            if env.rewards[human_agent] == 1:
                print("恭喜你赢了！")
            elif env.rewards[ai_agent] == 1:
                print("AI 获胜！")
            else:
                print("平局！")
            game_over = True

    # 等待片刻后退出
    pygame.time.wait(1000)


if __name__ == "__main__":
    human_vs_ai_game()