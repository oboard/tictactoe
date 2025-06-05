import os
import re
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
    def __init__(self, epsilon=0.6, count=30000):
        self.epsilon = epsilon        # 探索率
        self.count = count            # 训练回合数
        self.filename = os.path.join(dirname, "model_mcts.pkl")
        self.table = self.load()      # 状态-动作值表
        logger.info("load MCTS state count %s", len(self.table))

    def save(self):
        with open(self.filename, 'wb') as file:
            file.write(pickle.dumps(self.table))

    def load(self) -> dict[tuple, np.ndarray]:
        if not os.path.exists(self.filename):
            logger.info("No File!")
            return {}
        with open(self.filename, 'rb') as file:
            return pickle.loads(file.read())

    def end_game(self, state: np.ndarray) -> np.ndarray | None:
        r0 = list(np.sum(state, axis=0))  # 每一列的总和
        r1 = list(np.sum(state, axis=1))  # 每一行的总和
        r2 = [np.trace(state)]            # 主对角线的总和
        r3 = [np.trace(np.flip(state, axis=1))]  # 副对角线的总和
        r = r0 + r1 + r2 + r3             # 合并所有可能的胜利线

        # 三个数分别表示 黑，平，白
        if 3 in r:
            return np.array([1, 0, 0])
        if -3 in r:
            return np.array([0, 0, 1])
        if len(np.argwhere(state == 0)) == 0:
            return np.array([0, 1, 0])
        return None

    def hash(self, state: np.ndarray):
        return tuple(state.reshape(9))

    def act(self, state: np.ndarray, turn: int):
        # wheres = np.argwhere(state == EMPTY)
        # where = random.choice(wheres)
        # return tuple(where)
        return self.exploitation(state, turn)

    def exploration(self, state: np.ndarray):
        wheres = np.argwhere(state == EMPTY)
        assert (len(wheres) > 0)
        where = random.choice(wheres)
        return tuple(where), 0.0


    def exploitation(self, state: np.ndarray, turn: int):
        """选择最有价值的动作"""
        wheres = np.argwhere(state == EMPTY)
        assert (len(wheres) > 0)

        results = []
        state_key = self.hash(state)
        if state_key in self.table:
            values = self.table[state_key]
            for where in wheres:
                pos = where[0] * 3 + where[1]
                value = values[pos] if len(values) > pos else 0
                results.append((tuple(where), value))
        
        if not results:
            return self.exploration(state)
            
        # 选择最大值动作
        best_move = max(results, key=lambda x: x[1])
        return best_move

    def step(self, state: np.ndarray, turn: int, chain: list):
        if random.random() < self.epsilon:
            where, confidence = self.exploration(state)
        else:
            where, confidence = self.exploitation(state, turn)

        state[where] = turn
        chain.append(self.hash(state))
        end = self.end_game(state)
        if end is None:
            return self.step(state, turn * -1, chain)
        for key in chain:
            self.table.setdefault(key, np.array([0, 0, 0]))
            self.table[key] += end
        return

    def train(self, callback=None):
        """训练模型"""
        for i in tqdm(range(self.count)):
            state = np.zeros((3, 3), dtype=np.int8)  # 初始棋盘
            history = []  # 记录本局的移动历史
            turn = BLACK

            # 每100步更新一次训练窗口信息
            if callback and i % 100 == 0:
                callback(i, self.table, self.epsilon)

            # 模拟一局游戏
            while True:
                # 根据epsilon-greedy策略选择动作
                if random.random() < self.epsilon:
                    where, _ = self.exploration(state)
                else:
                    where, _ = self.exploitation(state, turn)

                # 记录状态和动作
                history.append((self.hash(state), where))
                
                # 执行动作
                state[where] = turn
                result = self.end_game(state)
                
                if result is not None:  # 游戏结束
                    # 更新经历的所有状态的价值
                    for state_key, action in history:
                        if state_key not in self.table:
                            self.table[state_key] = np.zeros(9, dtype=np.float32)
                            
                        action_idx = action[0] * 3 + action[1]
                        if result[0] == 1:  # 黑胜
                            reward = 1.0 if turn == BLACK else -1.0
                        elif result[2] == 1:  # 白胜
                            reward = 1.0 if turn == WHITE else -1.0
                        else:  # 平局
                            reward = 0.0
                        
                        # 更新动作值
                        self.table[state_key][action_idx] += reward
                    break

                turn *= -1  # 切换玩家

        # 保存模型
        self.save()
        
        # 训练结束时的最后一次回调
        if callback:
            callback(self.count, self.table, self.epsilon)
