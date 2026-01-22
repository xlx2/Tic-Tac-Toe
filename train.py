import random
from collections import defaultdict
from tictactoe import TicTacToeEnv
from utils import load_model, save_model, get_valid_actions


def train_agents(
        num_episodes: int,  # 训练总回合数
        alpha: float,  # 学习率
        gamma: float,  # 折扣因子
        epsilon_start: float,  # 初始探索率
        epsilon_end: float,  # 终止探索率
        epsilon_decay: float,  # 探索率衰减因子
        save_interval: int,  # 输出日志保存模型
        continue_training: bool  # 是否继续训练
    ) -> None:
    """训练两个 Q-Learning 智能体"""
    # 初始化两个智能体的Q表
    if continue_training:
        try:
            q_tables = load_model("final_q_tables.pkl")
            print("加载已有模型继续训练")
        except FileNotFoundError:
            print("未找到模型文件，开始新训练")
            q_tables = {
                "player_1": defaultdict(float),
                "player_2": defaultdict(float)
            }
    else:
        q_tables = {
            "player_1": defaultdict(float),
            "player_2": defaultdict(float)
        }
    # 初始化超参数
    epsilon = epsilon_start
    env = TicTacToeEnv()
    # 统计变量
    wins_player1 = 0
    wins_player2 = 0
    draws = 0

    print("开始训练...")
    for episode in range(num_episodes):
        env.reset()
        # 记录每轮的历史
        history = {
            "player_1": {"states": [], "actions": []},
            "player_2": {"states": [], "actions": []}
        }
        for agent in env.agent_iter():
            obs = env.observe(agent)
            state = tuple(obs)
            # 获取合法动作
            valid_actions = get_valid_actions((state, None))
            # Epsilon-greedy策略选择动作
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                # 选择当前状态下Q值最大的动作
                q_values = [q_tables[agent][(state, a)] for a in valid_actions]
                max_q = max(q_values) if q_values else 0
                best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
                action = random.choice(best_actions)
            # 记录历史
            history[agent]["states"].append(state)
            history[agent]["actions"].append(action)
            # 执行动作
            env.step(action)
            # 获取奖励和终止状态
            reward = env.rewards[agent]
            done = env.terminations[agent]
            # 游戏结束，更新Q表
            if done:
                # 更新当前智能体的Q值（最终奖励）
                if len(history[agent]["states"]) > 0:
                    last_state = history[agent]["states"][-1]
                    last_action = history[agent]["actions"][-1]
                    
                    current_q = q_tables[agent][(last_state, last_action)]
                    target_q = reward  # 游戏结束，没有未来奖励
                    new_q = current_q + alpha * (target_q - current_q)
                    q_tables[agent][(last_state, last_action)] = new_q

                # 更新对手的Q值
                opponent = "player_2" if agent == "player_1" else "player_1"
                if len(history[opponent]["states"]) > 0:
                    last_state = history[opponent]["states"][-1]
                    last_action = history[opponent]["actions"][-1]
                    
                    current_q = q_tables[opponent][(last_state, last_action)]
                    target_q = -reward  # 对手获得相反的奖励
                    new_q = current_q + alpha * (target_q - current_q)
                    q_tables[opponent][(last_state, last_action)] = new_q

                # 统计结果
                if reward == 1:
                    if agent == "player_1":
                        wins_player1 += 1
                    else:
                        wins_player2 += 1
                elif reward == -1:
                    if agent == "player_1":
                        wins_player2 += 1
                    else:
                        wins_player1 += 1
                else:
                    draws += 1
                break
            else:
                # 游戏未结束，更新Q值
                if len(history[agent]["states"]) > 1:
                    # 获取上一步的状态和动作
                    prev_state = history[agent]["states"][-2]
                    prev_action = history[agent]["actions"][-2]
                    
                    # 获取当前状态和可用动作
                    current_state = history[agent]["states"][-1]
                    current_valid_actions = get_valid_actions((current_state, None))
                    
                    # 计算目标Q值
                    if current_valid_actions:
                        current_q_values = [q_tables[agent][(current_state, a)] for a in current_valid_actions]
                        max_current_q = max(current_q_values) if current_q_values else 0
                    else:
                        max_current_q = 0
                    
                    target_q = reward + gamma * max_current_q
                    current_q = q_tables[agent][(prev_state, prev_action)]
                    new_q = current_q + alpha * (target_q - current_q)
                    q_tables[agent][(prev_state, prev_action)] = new_q

        # 更新探索率
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay

        # 输出日志
        if (episode + 1) % save_interval == 0:
            total = max(wins_player1 + wins_player2 + draws, 1)
            print(f"Episode {episode + 1}")
            print(f"Player 1 胜率: {wins_player1 / total * 100:.2f}%")
            print(f"Player 2 胜率: {wins_player2 / total * 100:.2f}%")
            print(f"平局率: {draws / total * 100:.2f}%")
            print(f"当前探索率: {epsilon:.4f}")
            print("-" * 40)

            # 重置统计
            wins_player1 = 0
            wins_player2 = 0
            draws = 0

        # 保存中间模型
        if (episode + 1) % save_interval == 0:
            save_model(q_tables, f"q_tables_episode_{episode+1}.pkl")

    # 保存最终模型
    save_model(q_tables, "final_q_tables.pkl")
    print("训练完成！最终模型已保存。")


if __name__ == "__main__":      
    q_tables = train_agents(
        num_episodes=100_000,
        alpha=0.1,
        gamma=1.0,
        epsilon_start=1.0,
        epsilon_end=0.001,
        epsilon_decay=0.9999,
        save_interval=10_000,
        continue_training=False
        )