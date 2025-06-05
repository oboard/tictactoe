import os
import random
import pickle
import numpy as np
from tqdm import tqdm
from logger import logger

dirname = os.path.dirname(os.path.abspath(__file__))

EMPTY = 0
BLACK = 1
WHITE = -1

class Model(object):
    def __init__(self, epsilon=0.6, alpha=0.5, gamma=0.9, count=30000):
        self.epsilon = epsilon        # 探索率
        self.alpha = alpha            # 学习率
        self.gamma = gamma            # 折扣因子
        self.count = count            # 训练回合数
        self.filename = os.path.join(dirname, "model_qlearning.pkl")
        self.Q = self.load()          # Q表：key=状态, value=动作Q值数组
        logger.info("load Q-learning state count %s", len(self.Q))

    def save(self):
        with open(self.filename, 'wb') as file:
            pickle.dump(self.Q, file)

    def load(self):
        if not os.path.exists(self.filename):
            logger.info("No Q-learning model file found!")
            return {}
        with open(self.filename, 'rb') as file:
            return pickle.load(file)

    def end_game(self, state: np.ndarray):
        r0 = list(np.sum(state, axis=0))
        r1 = list(np.sum(state, axis=1))
        r2 = [np.trace(state)]
        r3 = [np.trace(np.flip(state, axis=1))]
        r = r0 + r1 + r2 + r3
        if 3 in r:
            return np.array([1, 0, 0])  # 黑胜
        if -3 in r:
            return np.array([0, 0, 1])  # 白胜
        if len(np.argwhere(state == 0)) == 0:
            return np.array([0, 1, 0])  # 平局
        return None

    def hash(self, state: np.ndarray):
        return tuple(state.reshape(9))

    def available_moves(self, state):
        return [tuple(x) for x in np.argwhere(state == EMPTY)]

    def select_action(self, state, turn):
        """epsilon-greedy选动作：大部分时间选Q值最大，少部分探索"""
        state_key = self.hash(state)
        moves = self.available_moves(state)
        if (random.random() < self.epsilon) or (state_key not in self.Q):
            # 随机探索
            move = random.choice(moves)
        else:
            # 选择Q值最大的动作
            q_values = self.Q[state_key]
            valid_q = [q_values[move[0]*3 + move[1]] for move in moves]
            move = moves[np.argmax(valid_q)]
        return move

    def act(self, state, turn):
        """对弈时的行动（全贪婪）"""
        state_key = self.hash(state)
        moves = self.available_moves(state)
        if state_key not in self.Q:
            return random.choice(moves), 0.0
        q_values = self.Q[state_key]
        valid_q = [q_values[move[0]*3 + move[1]] for move in moves]
        move = moves[np.argmax(valid_q)]
        return move, np.max(valid_q)

    def train(self, callback=None):
        for i in tqdm(range(self.count)):
            state = np.zeros((3, 3), dtype=np.int8)
            turn = BLACK
            states = []
            actions = []
            
            # 每100步更新一次训练窗口信息
            if callback and i % 100 == 0:
                callback(i, self.Q, self.epsilon)
                
            while True:
                state_key = self.hash(state)
                moves = self.available_moves(state)
                if state_key not in self.Q:
                    self.Q[state_key] = np.zeros(9, dtype=np.float32)
                action = self.select_action(state, turn)
                actions.append((state_key, action, turn))
                state = state.copy()
                state[action] = turn
                states.append(state_key)
                result = self.end_game(state)
                if result is not None:
                    # 游戏结束，更新最后一步
                    winner = result.argmax()  # 0=黑胜, 1=平局, 2=白胜
                    if winner == 0:  # 黑胜
                        reward = 1.0 * turn
                    elif winner == 2:  # 白胜
                        reward = -1.0 * turn
                    else:
                        reward = 0.0
                    self.Q[state_key][action[0] * 3 + action[1]] += self.alpha * (
                                reward - self.Q[state_key][action[0] * 3 + action[1]])
                    break

                # 计算下一状态最大Q值
                next_key = self.hash(state)
                if next_key not in self.Q:
                    self.Q[next_key] = np.zeros(9, dtype=np.float32)
                # Q-Learning核心更新公式
                q = self.Q[state_key][action[0] * 3 + action[1]]
                next_q = self.Q[next_key]
                max_next_q = np.max(next_q)
                # 没有reward时，reward为0
                self.Q[state_key][action[0] * 3 + action[1]] += self.alpha * (self.gamma * max_next_q - q)

                turn *= -1  # 交换回合

        self.save()
        # 训练结束时的最后一次回调
        if callback:
            callback(self.count, self.Q, self.epsilon)
