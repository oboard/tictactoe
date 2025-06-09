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

# 井字棋特殊棋型定义
FORK_PATTERN = 1        # 叉子棋型：创造两路获胜机会
BLOCK_FORK_PATTERN = 2  # 阻挡叉子
CENTER_PATTERN = 3      # 中心占据
CORNER_PATTERN = 4      # 角落占据
OPPOSITE_CORNER = 5     # 对角角落
EDGE_PATTERN = 6        # 边缘占据


class Model(object):

    def __init__(self, epsilon=0.7, count=10000) -> None:
        self.epsilon = epsilon
        self.count = count
        self.filename = os.path.join(dirname, "model_default.pkl")
        self.table: dict[tuple, np.ndarray] = self.load()
        self.stats = {"black_wins": 0, "white_wins": 0, "draws": 0}  # 训练统计
        self.pattern_stats = {          # 记录各种棋型的识别次数
            FORK_PATTERN: 0,
            BLOCK_FORK_PATTERN: 0,
            CENTER_PATTERN: 0,
            CORNER_PATTERN: 0,
            OPPOSITE_CORNER: 0,
            EDGE_PATTERN: 0
        }
        logger.info("load model state count %s", self.table.__len__())

    def save(self):
        data = {
            'table': self.table,
            'stats': self.stats,
            'pattern_stats': self.pattern_stats
        }
        with open(self.filename, 'wb') as file:
            file.write(pickle.dumps(data))

    def load(self) -> dict[tuple, np.ndarray]:
        if not os.path.exists(self.filename):
            return {}
        with open(self.filename, 'rb') as file:
            data = pickle.loads(file.read())
            if isinstance(data, dict) and 'table' in data:
                self.table = data['table']
                self.stats = data.get('stats', {"black_wins": 0, "white_wins": 0, "draws": 0})
                self.pattern_stats = data.get('pattern_stats', {pattern: 0 for pattern in range(1, 7)})
                return self.table
            else:
                # 兼容旧版本
                self.stats = {"black_wins": 0, "white_wins": 0, "draws": 0}
                self.pattern_stats = {pattern: 0 for pattern in range(1, 7)}
                return data

    def end_game(self, state: np.ndarray) -> np.ndarray | None:
        r0 = list(np.sum(state, axis=0))
        r1 = list(np.sum(state, axis=1))
        r2 = [np.trace(state)]
        r3 = [np.trace(np.flip(state, axis=1))]
        r = r0 + r1 + r2 + r3

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
        wheres = np.argwhere(state == EMPTY)
        assert (len(wheres) > 0)

        results = []
        for where in wheres:
            where = tuple(where)
            s = state.copy()
            s[where] = turn

            key = self.hash(s)
            if key not in self.table:
                continue

            black, draw, white = self.table[key]
            p = (black - white) / sum(self.table[key]) * turn
            results.append((where, p))

        if not results:
            return self.exploration(state)

        result = sorted(results, key=lambda e: e[1])[-1]
        return result
        
    def detect_patterns(self, state: np.ndarray, turn: int):
        """检测棋盘中的特殊棋型并更新统计"""
        # 1. 检测中心占据
        if state[1, 1] == turn:
            self.pattern_stats[CENTER_PATTERN] += 1
            
        # 2. 检测角落占据
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        corner_values = [state[x, y] for x, y in corners]
        if turn in corner_values:
            self.pattern_stats[CORNER_PATTERN] += 1
            
        # 3. 检测叉子棋型
        if self.detect_fork_threat(state, turn):
            self.pattern_stats[FORK_PATTERN] += 1
            
        # 4. 检测阻挡叉子
        if self.detect_fork_threat(state, -turn):
            self.pattern_stats[BLOCK_FORK_PATTERN] += 1
            
        # 5. 检测对角角落策略
        if self.is_opposite_corner_play(state, turn):
            self.pattern_stats[OPPOSITE_CORNER] += 1
            
        # 6. 检测边缘占据
        if self.is_edge_play(state, turn):
            self.pattern_stats[EDGE_PATTERN] += 1
            
    def detect_fork_threat(self, state: np.ndarray, player: int):
        """检测叉子棋型（同时有两路获胜的威胁）"""
        # 复制当前状态用于测试
        test_state = state.copy()
        winning_moves = []
        
        # 尝试每个空位
        empty_positions = [(i, j) for i in range(3) for j in range(3) if state[i, j] == 0]
        for move in empty_positions:
            test_state[move] = player
            
            # 检查行
            for i in range(3):
                row_sum = np.sum(test_state[i, :])
                if row_sum == 2 * player and 0 in test_state[i, :]:
                    winning_moves.append(move)
                    break
                    
            # 检查列
            for i in range(3):
                col_sum = np.sum(test_state[:, i])
                if col_sum == 2 * player and 0 in test_state[:, i]:
                    winning_moves.append(move)
                    break
                    
            # 检查对角线
            diag_sum = np.trace(test_state)
            if diag_sum == 2 * player and 0 in np.diag(test_state):
                winning_moves.append(move)
                
            anti_diag_sum = np.trace(np.flip(test_state, axis=1))
            if anti_diag_sum == 2 * player and 0 in np.diag(np.flip(test_state, axis=1)):
                winning_moves.append(move)
                
            # 恢复状态用于下一次测试
            test_state[move] = 0
        
        # 如果有两个或更多不同的获胜威胁，就是叉子棋型
        return len(set(winning_moves)) >= 2
        
    def is_opposite_corner_play(self, state: np.ndarray, player: int):
        """检测是否形成对角角落策略"""
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        opposite_corners = {
            (0, 0): (2, 2),
            (0, 2): (2, 0),
            (2, 0): (0, 2),
            (2, 2): (0, 0)
        }
        
        for corner in corners:
            if state[corner] == -player:  # 对手占据角落
                opp_corner = opposite_corners[corner]
                if state[opp_corner] == player:  # 我方占据对角角落
                    return True
        return False
        
    def is_edge_play(self, state: np.ndarray, player: int):
        """检测是否形成边缘策略"""
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
        center_free = state[1, 1] == 0
        corners_free = all(state[c] == 0 for c in [(0, 0), (0, 2), (2, 0), (2, 2)])
        
        for edge in edges:
            if state[edge] == player:
                # 如果中心和角落都不可用，则边缘是次优选择
                if not center_free and not corners_free:
                    return True
        return False

    def step(self, state: np.ndarray, turn: int, chain: list):
        if random.random() < self.epsilon:
            where, confidence = self.exploration(state)
        else:
            where, confidence = self.exploitation(state, turn)

        state[where] = turn
        # 检测棋型并更新统计
        self.detect_patterns(state, turn)
        
        chain.append(self.hash(state))
        end = self.end_game(state)
        if end is None:
            return self.step(state, turn * -1, chain)
            
        # 更新胜率统计
        if end[0] == 1:  # 黑胜
            self.stats["black_wins"] += 1
        elif end[2] == 1:  # 白胜
            self.stats["white_wins"] += 1
        else:  # 平局
            self.stats["draws"] += 1
            
        for key in chain:
            self.table.setdefault(key, np.array([0, 0, 0]))
            self.table[key] += end
        return

    def train(self, callback=None):
        state = np.zeros((3, 3), dtype=np.int8)
        turn = BLACK
        bar = tqdm(range(self.count))
        for i in bar:
            self.step(state.copy(), turn, [])
            bar.set_postfix(cnt=len(self.table))
            
            # 每100步更新一次训练窗口信息
            if callback and i % 100 == 0:
                callback(i, self.table, self.epsilon)
                
            # 每1000局保存一次模型
            if i > 0 and i % 1000 == 0:
                self.save()
                
            # 定期显示训练统计
            if i > 0 and i % 5000 == 0:
                total = self.stats["black_wins"] + self.stats["white_wins"] + self.stats["draws"]
                if total > 0:
                    logger.info(f"默认模型训练进度 {i}/{self.count} - 状态数: {len(self.table)} - "
                             f"黑胜率: {self.stats['black_wins']/total:.2f} - "
                             f"白胜率: {self.stats['white_wins']/total:.2f} - "
                             f"平局率: {self.stats['draws']/total:.2f}")
                
        # 训练结束后自动保存
        self.save()
        
        # 训练结束时的最后一次回调
        if callback:
            callback(self.count, self.table, self.epsilon)
            
        # 显示最终训练统计
        total = self.stats["black_wins"] + self.stats["white_wins"] + self.stats["draws"]
        if total > 0:
            logger.info(f"默认模型训练完成 - 状态数: {len(self.table)} - "
                      f"黑胜率: {self.stats['black_wins']/total:.2f} - "
                      f"白胜率: {self.stats['white_wins']/total:.2f} - "
                      f"平局率: {self.stats['draws']/total:.2f}")
