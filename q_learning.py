import random
import pickle
from collections import defaultdict


class QLearningAgent:
    """
    Q Learning algorithm
    """
    def __init__(self, alpha, gamma, epsilon):
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.q_table = defaultdict(float)  # Q表，默认值为0

    def get_action(self, state, valid_actions, training=True):
        """ 根据当前状态选择动作 """
        # 将valid_actions转换为list，确保兼容numpy数组
        valid_actions_list = list(valid_actions) if hasattr(valid_actions, '__iter__') else valid_actions
        # epsilon greedy
        if training and random.random() < self.epsilon:
            # 探索：随机选择有效动作
            return random.choice(valid_actions_list)
        # 利用：选择Q值最高的动作
        q_values = [self.q_table[(state, a)] for a in valid_actions_list]
        max_q = max(q_values) if q_values else 0
        # 如果有多个动作具有相同的最大Q值，随机选择一个
        best_actions = [a for a, q in zip(valid_actions_list, q_values) if q == max_q]

        return random.choice(best_actions) if best_actions else random.choice(valid_actions_list)

    def update(self, state, action, reward, next_state, next_valid_actions):
        """ 更新Q值 """
        current_q = self.q_table[(state, action)]
        # 计算未来奖励的最大值
        next_q_values = [self.q_table[(next_state, a)] for a in next_valid_actions]
        max_next_q = max(next_q_values) if next_q_values else 0
        # Q学习更新规则
        target_q = reward + self.gamma * max_next_q
        new_q = current_q + self.alpha * (target_q - current_q)
        
        self.q_table[(state, action)] = new_q

    def save_model(self, filepath):
        """ 保存 Q 表 """
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"模型已保存至 {filepath}")
    
    def load_model(self, filepath):
        """加载 Q 表"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        q = defaultdict(float)
        q.update(data)
        self.q_table = q