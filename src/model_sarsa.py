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
        # 新增：开局策略统计
        self.opening_stats = {
            'center': {'wins': 0, 'total': 0},
            'corner': {'wins': 0, 'total': 0},
            'edge': {'wins': 0, 'total': 0}
        }
        logger.info("load SARSA state count %s", len(self.Q))

    def save(self):
        data = {
            'Q': self.Q,
            'stats': self.stats,
            'pattern_stats': self.pattern_stats,
            'opening_stats': self.opening_stats  # 新增：保存开局统计
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
                self.opening_stats = data.get('opening_stats', {
                    'center': {'wins': 0, 'total': 0},
                    'corner': {'wins': 0, 'total': 0},
                    'edge': {'wins': 0, 'total': 0}
                })
                return self.Q
            else:
                # 兼容旧版本
                self.stats = {"black_wins": 0, "white_wins": 0, "draws": 0}
                self.pattern_stats = {pattern: 0 for pattern in range(1, 7)}
                self.opening_stats = {
                    'center': {'wins': 0, 'total': 0},
                    'corner': {'wins': 0, 'total': 0},
                    'edge': {'wins': 0, 'total': 0}
                }
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
        reward = 0.0
        
        # 一、关键形态识别
        # 1. 检查行、列和对角线上连续两子的情况，并判断是否为防守动作
        last_move = np.argwhere(state == turn)[-1] if len(np.argwhere(state == turn)) > 0 else None
        if last_move is not None:
            last_row, last_col = last_move

            # 检查是否为防守动作
            is_defensive_move = False
            
            # 检查行
            for i in range(3):
                row = state[i, :]
                row_sum = np.sum(row)
                if abs(row_sum) == 2 and 0 in row:  # 两子一空的情况
                    empty_pos = np.where(row == 0)[0][0]
                    if np.sign(row_sum) == -turn and i == last_row and empty_pos == last_col:  # 是防守动作
                        reward += 5.0 * turn
                        is_defensive_move = True
                        break
                        
            # 检查列
            if not is_defensive_move:
                for i in range(3):
                    col = state[:, i]
                    col_sum = np.sum(col)
                    if abs(col_sum) == 2 and 0 in col:
                        empty_pos = np.where(col == 0)[0][0]
                        if np.sign(col_sum) == -turn and empty_pos == last_row and i == last_col:  # 是防守动作
                            reward += 5.0 * turn
                            is_defensive_move = True
                            break
                            
            # 检查主对角线
            if not is_defensive_move:
                diag = np.diag(state)
                diag_sum = np.sum(diag)
                if abs(diag_sum) == 2 and 0 in diag:
                    empty_pos = np.where(diag == 0)[0][0]
                    if np.sign(diag_sum) == -turn and last_row == empty_pos and last_col == empty_pos:  # 是防守动作
                        reward += 5.0 * turn
                        is_defensive_move = True
                        
            # 检查副对角线
            if not is_defensive_move:
                anti_diag = np.diag(np.flip(state, axis=1))
                anti_diag_sum = np.sum(anti_diag)
                if abs(anti_diag_sum) == 2 and 0 in anti_diag:
                    empty_pos = np.where(anti_diag == 0)[0][0]
                    if np.sign(anti_diag_sum) == -turn and last_row == empty_pos and last_col == 2 - empty_pos:  # 是防守动作
                        reward += 5.0 * turn
                        is_defensive_move = True
            
        # 其他奖励计算...
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
        # 重置棋型统计
        self.pattern_stats = {
            FORK_PATTERN: 0,
            BLOCK_FORK_PATTERN: 0,
            CENTER_PATTERN: 0,
            CORNER_PATTERN: 0,
            OPPOSITE_CORNER: 0,
            EDGE_PATTERN: 0
        }
        # 重置胜率统计
        self.stats = {"black_wins": 0, "white_wins": 0, "draws": 0}
        
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
            first_move = action  # 记录第一步落子
            
            # 记录上一个状态是否有对方连子威胁
            prev_state_threatened = False
            
            while True:
                # 检查当前状态是否有对方连子威胁
                current_threatened = False
                test_state = state.copy()
                for test_move in self.available_moves(test_state):
                    test_state[test_move] = -turn
                    if self.end_game(test_state) is not None:
                        current_threatened = True
                        break
                    test_state[test_move] = 0
                
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
                            self.update_opening_stats(first_move, True)
                        else:
                            self.stats["white_wins"] += 1
                    elif winner == 2:  # 白胜
                        # 如果之前有威胁没防守导致输掉，额外惩罚
                        if prev_state_threatened:
                            reward = -5.0 * turn
                        else:
                            reward = -1.0 * turn
                        if turn == WHITE:
                            self.stats["white_wins"] += 1
                        else:
                            self.stats["black_wins"] += 1
                            self.update_opening_stats(first_move, False)
                    else:  # 平局
                        reward = 0.0
                        self.stats["draws"] += 1
                        self.update_opening_stats(first_move, 0.5)
                        
                    # 终局直接更新
                    self.Q[state_key][action_idx] += self.alpha * (
                        reward - self.Q[state_key][action_idx])
                    break

                # 获取中间奖励
                intermediate_reward = self.get_reward_shaping(state_, turn)
                
                # 更新威胁状态
                prev_state_threatened = current_threatened
                
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
                
            # 每1000局保存一次模型，并更新开局策略
            if i > 0 and i % 1000 == 0:
                self.adjust_opening_q_values()  # 根据统计调整开局Q值
                self.save()
                
            # 定期显示训练统计
            if i > 0 and i % 5000 == 0:
                total = self.stats["black_wins"] + self.stats["white_wins"] + self.stats["draws"]
                if total > 0:
                    # 显示开局策略统计
                    logger.info(f"SARSA训练进度 {i}/{self.count} - 状态数: {len(self.Q)} - "
                             f"黑胜率: {self.stats['black_wins']/total:.2f} - "
                             f"白胜率: {self.stats['white_wins']/total:.2f} - "
                             f"平局率: {self.stats['draws']/total:.2f}")
                    logger.info("开局策略统计:")
                    for move_type in ['center', 'corner', 'edge']:
                        stats = self.opening_stats[move_type]
                        win_rate = stats['wins'] / stats['total'] if stats['total'] > 0 else 0
                        logger.info(f"{move_type}: 胜率 {win_rate:.2f} (总计: {stats['total']})")
                
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

    def get_move_type(self, move):
        """判断落子类型（中心、角落、边缘）"""
        if move == (1, 1):
            return 'center'
        if move in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            return 'corner'
        return 'edge'

    def update_opening_stats(self, first_move, is_win):
        """更新开局策略统计"""
        move_type = self.get_move_type(first_move)
        self.opening_stats[move_type]['total'] += 1
        if is_win:
            self.opening_stats[move_type]['wins'] += 1

    def adjust_opening_q_values(self):
        """根据开局统计调整Q值"""
        empty_state = np.zeros((3, 3), dtype=np.int8)
        state_key = self.hash(empty_state)
        if state_key not in self.Q:
            self.Q[state_key] = np.zeros(9, dtype=np.float32)

        # 计算各类型的胜率
        win_rates = {}
        for move_type in ['center', 'corner', 'edge']:
            stats = self.opening_stats[move_type]
            win_rate = stats['wins'] / stats['total'] if stats['total'] > 0 else 0.33
            win_rates[move_type] = win_rate

        # 根据胜率调整Q值
        for i in range(3):
            for j in range(3):
                move_type = self.get_move_type((i, j))
                self.Q[state_key][i*3 + j] = win_rates[move_type]
