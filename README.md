# 井字棋强化学习项目

这是一个基于强化学习的井字棋（Tic-Tac-Toe）游戏项目，实现了多智能体环境下的Q-Learning算法。

## 项目特点

- 完整的多智能体强化学习环境
- 支持可视化界面的人机对战
- 包含训练、评估和对战功能
- 支持模型持久化保存与加载

## 文件结构

```
tictactoe/
├── tictactoe.py      # 强化学习环境定义
├── q_learning.py     # Q学习算法实现
├── train.py          # 训练脚本
├── evaluate.py       # 评估脚本
├── human_play.py     # 可视化人机对战
├── utils.py          # 工具函数
├── img/              # 图像资源目录
│   ├── board.png     # 棋盘背景图
│   ├── X.png         # X棋子图
│   ├── O.png         # O棋子图
│   └── tictactoe.png # 应用图标
└── README.md         # 项目说明文档
```

## 环境依赖

- Python 3.11
- NumPy
- PyGame
- PettingZoo
- Gymnasium

安装依赖包：
```bash
pip install numpy pygame pettingzoo gymnasium
```

## 使用方法

### 1. 训练模型

```bash
python train.py
```

此命令将启动两个AI代理之间的训练过程，它们会自我对弈以提升技能。训练完成后会保存Q表模型。

您可以修改以下参数来自定义训练：

- `num_episodes`: 训练的回合数
- `lr`: 学习率
- `gamma`: 折扣因子
- `epsilon`: 探索率初始值
- `epsilon_decay`: 探索率衰减率
- `min_epsilon`: 最小探索率

### 2. 可视化人机对战

```bash
python human_play.py
```

此命令将启动带图形界面的人机对战，玩家可以通过鼠标点击棋盘来下棋。


## 项目架构

### 强化学习环境 (tictactoe.py)

- 使用PettingZoo框架构建多智能体环境
- 集成了Pygame图形界面渲染功能
- 实现了完整的井字棋规则

### Q学习算法 (q_learning.py)

- 实现了多智能体Q学习算法
- 支持ε-贪婪策略进行探索
- 包含渐进式探索率衰减

### 训练流程 (train.py)

- 控制模型训练过程
- 定期保存模型快照
- 包含训练进度监控

## 游戏规则

- 两名玩家轮流在3x3的棋盘上下棋
- 先形成三连（横、竖、对角线）的玩家获胜
- 棋盘填满且无人获胜则为平局

## 技术细节

- 状态空间：3^9 (每个格子可为X、O或空)
- 动作空间：离散空间，0-8对应棋盘的9个位置
- 奖励机制：获胜+1，失败-1，平局0
- 观察空间：长度为9的向量，-1表示O，0表示空，1表示X


