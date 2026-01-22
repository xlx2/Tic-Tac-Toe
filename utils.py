import pickle
from collections import defaultdict
import numpy as np


def save_model(q_table, filepath):
    """保存 Q 表"""
    with open(filepath, 'wb') as f:
        pickle.dump(dict(q_table), f)
    print(f"模型已保存至 {filepath}")

def load_model(filepath):
    """加载 Q 表"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    q = defaultdict(float)
    q.update(data)
    return q

def get_valid_actions(state):
    """获取当前状态下可用的动作列表"""
    board = np.array(state[0])  # 状态是一个元组，第一个元素是棋盘
    return [i for i in range(9) if board[i] == 0]