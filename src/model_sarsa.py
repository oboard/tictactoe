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

# 井字棋特殊棋型定义
FORK_PATTERN = 1        # 叉子棋型：创造两路获胜机会
BLOCK_FORK_PATTERN = 2  # 阻挡叉子
CENTER_PATTERN = 3      # 中心占据
CORNER_PATTERN = 4      # 角落占据
OPPOSITE_CORNER = 5     # 对角角落
EDGE_PATTERN = 6        # 边缘占据

class Model(object):
    def __init__(self, epsilon=0.6, alpha=0.5, gamma=0.9, count=30000):
        self.initial_epsilon = epsilon  # 初始探索率
        self.epsilon = epsilon          # 当前探索率
        self.epsilon_decay = 0.9995     # 探索率衰减系数
        self.epsilon_min = 0.01         # 最小探索率
        self.alpha = alpha              # 学习率
        self.gamma = gamma              # 折扣因子
        self.count = count              # 训练回合数
        self.filename = os.path.join(dirname, "model_sarsa.pkl")
        self.Q = self.load()            # Q表：key=状态, value=动作Q值数组
        self.stats = {"black_wins": 0, "white_wins": 0, "draws": 0}  # 训练统计
        self.pattern_stats = {          # 记录各种棋型的识别次数
            FORK_PATTERN: 0,
            BLOCK_FORK_PATTERN: 0,
            CENTER_PATTERN: 0,
            CORNER_PATTERN: 0,
            OPPOSITE_CORNER: 0,
            EDGE_PATTERN: 0
        }
        logger.info("load SARSA state count %s", len(self.Q))

    def save(self):
        data = {
            'Q': self.Q,
            'stats': self.stats,
            'pattern_stats': self.pattern_stats
        }
        with open(self.filename, 'wb') as file:
            pickle.dump(data, file)

    def load(self):
        if not os.path.exists(self.filename):
            logger.info("No SARSA model file found!")
            return {}
        with open(self.filename, 'rb') as file:
            data = pickle.load(file)
            if isinstance(data, dict) and 'Q' in data:
                self.Q = data['Q']
                self.stats = data.get('stats', {"black_wins": 0, "white_wins": 0, "draws": 0})
                self.pattern_stats = data.get('pattern_stats', {pattern: 0 for pattern in range(1, 7)})
                return self.Q
            else:
                # 兼容旧版本
                self.stats = {"black_wins": 0, "white_wins": 0, "draws": 0}
                self.pattern_stats = {pattern: 0 for pattern in range(1, 7)}
                return data

    def end_game(self, state: np.ndarray):
        r0 = list(np.sum(state, axis=0))
        r1 = list(np.sum(state, axis=1))
        r2 = [np.trace(state)]
        r3 = [np.trace(np.flip(state, axis=1))]
        r = r0 + r1 + r2 + r3
        if 3 in r:
            return np.array([1, 0, 0])    # 黑胜
        if -3 in r:
            return np.array([0, 0, 1])    # 白胜
        if len(np.argwhere(state == 0)) == 0:
            return np.array([0, 1, 0])    # 平局
        return None

    def get_state_symmetries(self, state: np.ndarray):
        """获取状态的所有对称形式"""
        symmetries = []
        # 原始状态
        symmetries.append(state)
        # 旋转90度
        symmetries.append(np.rot90(state))
        # 旋转180度
        symmetries.append(np.rot90(state, 2))
        # 旋转270度
        symmetries.append(np.rot90(state, 3))
        # 水平翻转
        symmetries.append(np.fliplr(state))
        # 垂直翻转
        symmetries.append(np.flipud(state))
        # 对角线翻转
        symmetries.append(np.transpose(state))
        # 反对角线翻转
        symmetries.append(np.fliplr(np.transpose(state)))
        return symmetries

    def hash(self, state: np.ndarray):
        """优化的状态哈希函数"""
        # 获取所有对称状态
        symmetries = self.get_state_symmetries(state)
        # 选择字典序最小的状态作为规范形式
        canonical_state = min(tuple(s.reshape(9)) for s in symmetries)
        return canonical_state

    def available_moves(self, state):
        return [tuple(x) for x in np.argwhere(state == EMPTY)]

    def get_reward_shaping(self, state: np.ndarray, turn: int) -> float:
        """增强版奖励塑形：根据当前状态给予中间奖励"""
        reward = 0.0
        
        # 一、关键形态识别
        # 1. 检查行、列和对角线上连续两子的情况
        for i in range(3):
            # 检查行
            row = state[i, :]
            row_sum = np.sum(row)
            if abs(row_sum) == 2 and 0 in row:  # 两子一空，可能下一步获胜
                reward += 0.2 * np.sign(row_sum) * turn
            elif abs(row_sum) == 1 and 0 in row and list(row).count(0) == 2:  # 一子两空，占据中间位置
                reward += 0.05 * np.sign(row_sum) * turn
                
            # 检查列
            col = state[:, i]
            col_sum = np.sum(col)
            if abs(col_sum) == 2 and 0 in col:
                reward += 0.2 * np.sign(col_sum) * turn
            elif abs(col_sum) == 1 and 0 in col and list(col).count(0) == 2:
                reward += 0.05 * np.sign(col_sum) * turn
                
        # 检查对角线
        diag = np.diag(state)
        diag_sum = np.sum(diag)
        if abs(diag_sum) == 2 and 0 in diag:
            reward += 0.2 * np.sign(diag_sum) * turn
        elif abs(diag_sum) == 1 and 0 in diag and list(diag).count(0) == 2:
            reward += 0.05 * np.sign(diag_sum) * turn
            
        anti_diag = np.diag(np.flip(state, axis=1))
        anti_diag_sum = np.sum(anti_diag)
        if abs(anti_diag_sum) == 2 and 0 in anti_diag:
            reward += 0.2 * np.sign(anti_diag_sum) * turn
        elif abs(anti_diag_sum) == 1 and 0 in anti_diag and list(anti_diag).count(0) == 2:
            reward += 0.05 * np.sign(anti_diag_sum) * turn
            
        # 2. 优先占据中心
        if state[1, 1] == turn:
            reward += 0.1
            self.pattern_stats[CENTER_PATTERN] += 1
        elif state[1, 1] == -turn:
            reward -= 0.1
            
        # 3. 优先占据角落比边缘
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        corner_values = [state[x, y] for x, y in corners]
        corners_own = corner_values.count(turn)
        corners_opp = corner_values.count(-turn)
        reward += 0.05 * corners_own - 0.05 * corners_opp
        
        if corners_own > 0:
            self.pattern_stats[CORNER_PATTERN] += 1
            
        # 三、井字棋特殊棋型识别与奖励
        # 1. 叉子棋型（同时有两路获胜的威胁）
        fork_threat = self.detect_fork_threat(state, turn)
        if fork_threat:
            reward += 0.3 * turn  # 高奖励，这是极强的棋型
            self.pattern_stats[FORK_PATTERN] += 1
            
        # 2. 对手的叉子棋型检测与阻挡
        opponent_fork = self.detect_fork_threat(state, -turn)
        if opponent_fork:
            reward -= 0.25 * turn  # 对手有叉子威胁是个坏局面
            # 这里的奖励略低于自己的叉子，确保进攻优先于防守
            self.pattern_stats[BLOCK_FORK_PATTERN] += 1
            
        # 3. 对角角落策略（如果对手占据一个角落，则占据对角角落）
        if self.is_opposite_corner_play(state, turn):
            reward += 0.1 * turn
            self.pattern_stats[OPPOSITE_CORNER] += 1
            
        # 4. 边缘占据策略（如果中心和角落不可用）
        if self.is_edge_play(state, turn):
            reward += 0.05 * turn
            self.pattern_stats[EDGE_PATTERN] += 1
            
        # 5. 开局策略奖励
        if np.count_nonzero(state) <= 3:  # 游戏开始阶段
            reward += self.opening_strategy_reward(state, turn)
            
        return reward
        
    def detect_fork_threat(self, state: np.ndarray, player: int):
        """检测叉子棋型（同时有两路获胜的威胁）"""
        # 复制当前状态用于测试
        test_state = state.copy()
        winning_moves = []
        
        # 尝试每个空位
        for move in self.available_moves(state):
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
        
    def opening_strategy_reward(self, state: np.ndarray, player: int):
        """开局策略奖励"""
        move_count = np.count_nonzero(state)
        reward = 0.0
        
        if move_count == 1:
            # 第一步走中心是最优策略
            if state[1, 1] == player:
                reward += 0.2
        elif move_count == 2:
            # 如果对手第一步走了中心，第二步走角落是好策略
            if state[1, 1] == -player:
                corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
                for corner in corners:
                    if state[corner] == player:
                        reward += 0.15
                        break
        elif move_count == 3:
            # 如果我方第一步走了中心，第三步应避免对手形成叉子
            if state[1, 1] == player:
                # 检查对手是否可以形成叉子威胁
                if not self.detect_fork_threat(state, -player):
                    reward += 0.1
                    
        return reward

    def select_action(self, state, turn):
        """epsilon-greedy选动作：大部分时间选Q值最大，少部分探索"""
        state_key = self.hash(state)
        moves = self.available_moves(state)
        if (random.random() < self.epsilon) or (state_key not in self.Q):
            move = random.choice(moves)
        else:
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
            
            # 更新探索率
            self.epsilon = max(self.epsilon_min, 
                             self.initial_epsilon * (self.epsilon_decay ** i))
            
            # 每100步更新一次训练窗口信息
            if callback and i % 100 == 0:
                callback(i, self.Q, self.epsilon)

            # 初始化第一个动作
            state_key = self.hash(state)
            if state_key not in self.Q:
                self.Q[state_key] = np.zeros(9, dtype=np.float32)
            action = self.select_action(state, turn)
            
            while True:
                # 落子
                state_ = state.copy()
                state_[action] = turn
                result = self.end_game(state_)
                action_idx = action[0]*3 + action[1]

                if result is not None:
                    winner = result.argmax()  # 0=黑胜, 1=平局, 2=白胜
                    if winner == 0:  # 黑胜
                        reward = 1.0 * turn
                        if turn == BLACK:
                            self.stats["black_wins"] += 1
                        else:
                            self.stats["white_wins"] += 1
                    elif winner == 2:  # 白胜
                        reward = -1.0 * turn
                        if turn == WHITE:
                            self.stats["white_wins"] += 1
                        else:
                            self.stats["black_wins"] += 1
                    else:  # 平局
                        reward = 0.0
                        self.stats["draws"] += 1
                        
                    # 终局直接更新
                    self.Q[state_key][action_idx] += self.alpha * (
                        reward - self.Q[state_key][action_idx])
                    break

                # 获取中间奖励
                intermediate_reward = self.get_reward_shaping(state_, turn)
                
                # 下一步状态和动作
                next_key = self.hash(state_)
                if next_key not in self.Q:
                    self.Q[next_key] = np.zeros(9, dtype=np.float32)
                next_action = self.select_action(state_, -turn)
                next_action_idx = next_action[0]*3 + next_action[1]
                
                # SARSA核心：用"实际选择的下一个动作"的Q值来更新
                q = self.Q[state_key][action_idx]
                next_q = self.Q[next_key][next_action_idx]
                self.Q[state_key][action_idx] += self.alpha * (
                    intermediate_reward + self.gamma * next_q - q)
                
                # 状态转移
                state = state_
                state_key = next_key
                action = next_action
                turn *= -1
                
            # 每1000局保存一次模型
            if i > 0 and i % 1000 == 0:
                self.save()
                
            # 定期显示训练统计
            if i > 0 and i % 5000 == 0:
                total = self.stats["black_wins"] + self.stats["white_wins"] + self.stats["draws"]
                if total > 0:
                    logger.info(f"SARSA训练进度 {i}/{self.count} - 状态数: {len(self.Q)} - "
                             f"黑胜率: {self.stats['black_wins']/total:.2f} - "
                             f"白胜率: {self.stats['white_wins']/total:.2f} - "
                             f"平局率: {self.stats['draws']/total:.2f}")
                
        self.save()
        # 训练结束时的最后一次回调
        if callback:
            callback(self.count, self.Q, self.epsilon)
            
        # 显示最终训练统计
        total = self.stats["black_wins"] + self.stats["white_wins"] + self.stats["draws"]
        if total > 0:
            logger.info(f"SARSA训练完成 - 状态数: {len(self.Q)} - "
                      f"黑胜率: {self.stats['black_wins']/total:.2f} - "
                      f"白胜率: {self.stats['white_wins']/total:.2f} - "
                      f"平局率: {self.stats['draws']/total:.2f}")
