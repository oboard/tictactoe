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
    def __init__(self, epsilon=0.3, alpha=0.1, gamma=0.95, count=100000):
        self.initial_epsilon = epsilon  # 初始探索率降低，更注重利用
        self.epsilon = epsilon          # 当前探索率
        self.epsilon_decay = 0.9999     # 探索率衰减系数更平缓
        self.epsilon_min = 0.01         # 最小探索率
        self.alpha = alpha              # 学习率降低，使学习更稳定
        self.gamma = gamma              # 折扣因子提高，更重视未来奖励
        self.count = count              # 训练回合数增加
        self.filename = os.path.join(dirname, "model_qlearning.pkl")
        self.Q = self.load()            # Q表：key=状态, value=动作Q值数组
        self.V = {}                     # 状态价值表：key=状态, value=状态价值
        self.win_states = {}            # 记录赢的状态
        self.lose_states = {}           # 记录输的状态
        self.stats = {"black_wins": 0, "white_wins": 0, "draws": 0}  # 训练统计
        self.pattern_stats = {          # 记录各种棋型的识别次数
            FORK_PATTERN: 0,
            BLOCK_FORK_PATTERN: 0,
            CENTER_PATTERN: 0,
            CORNER_PATTERN: 0,
            OPPOSITE_CORNER: 0,
            EDGE_PATTERN: 0
        }
        logger.info("load Q-learning state count %s", len(self.Q))

    def save(self):
        """保存模型"""
        # 将Q表和V表转换为可序列化的格式
        q_data = {str(k): v.tolist() for k, v in self.Q.items()}
        v_data = {str(k): v for k, v in self.V.items()}
        win_states = {str(k): v for k, v in self.win_states.items()}
        lose_states = {str(k): v for k, v in self.lose_states.items()}
        with open(self.filename, 'wb') as f:
            pickle.dump((q_data, v_data, win_states, lose_states, self.stats, self.pattern_stats), f)

    def load(self):
        """加载模型"""
        if not os.path.exists(self.filename):
            return {}
        try:
            with open(self.filename, 'rb') as f:
                data = pickle.load(f)
                if len(data) == 2:  # 兼容旧版本
                    q_data, v_data = data
                    self.Q = {eval(k): np.array(v, dtype=np.float32) for k, v in q_data.items()}
                    self.V = {eval(k): v for k, v in v_data.items()}
                    self.win_states = {}
                    self.lose_states = {}
                    self.stats = {"black_wins": 0, "white_wins": 0, "draws": 0}
                    self.pattern_stats = {pattern: 0 for pattern in range(1, 7)}
                elif len(data) == 5:  # 兼容中间版本
                    q_data, v_data, win_states, lose_states, self.stats = data
                    self.Q = {eval(k): np.array(v, dtype=np.float32) for k, v in q_data.items()}
                    self.V = {eval(k): v for k, v in v_data.items()}
                    self.win_states = {eval(k): v for k, v in win_states.items()}
                    self.lose_states = {eval(k): v for k, v in lose_states.items()}
                    self.pattern_stats = {pattern: 0 for pattern in range(1, 7)}
                else:
                    q_data, v_data, win_states, lose_states, self.stats, self.pattern_stats = data
                    self.Q = {eval(k): np.array(v, dtype=np.float32) for k, v in q_data.items()}
                    self.V = {eval(k): v for k, v in v_data.items()}
                    self.win_states = {eval(k): v for k, v in win_states.items()}
                    self.lose_states = {eval(k): v for k, v in lose_states.items()}
                return self.Q
        except:
            return {}

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
        return canonical_state  # 直接返回元组作为键

    def available_moves(self, state):
        return [tuple(x) for x in np.argwhere(state == EMPTY)]

    def get_reward_shaping(self, state: np.ndarray, turn: int) -> float:
        """增强版奖励塑形：根据当前状态给予中间奖励"""
        reward = 0.0
        
        # 一、胜负状态识别与加强奖励
        # 检查是否是必胜或必败状态
        state_key = self.hash(state)
        if state_key in self.win_states and turn == BLACK:
            return 0.3  # 黑方在必胜状态
        if state_key in self.lose_states and turn == BLACK:
            return -0.3  # 黑方在必败状态
        if state_key in self.win_states and turn == WHITE:
            return -0.3  # 白方在黑方必胜状态
        if state_key in self.lose_states and turn == WHITE:
            return 0.3  # 白方在黑方必败状态
        
        # 二、关键形态识别
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
            
        # 5. 阻挡对手获胜的关键点
        if self.is_blocking_win(state, turn):
            reward += 0.15 * turn
            
        # 6. 开局策略奖励
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
        
    def is_blocking_win(self, state: np.ndarray, player: int):
        """检测是否阻挡了对手的必胜点"""
        test_state = state.copy()
        for move in self.available_moves(state):
            # 尝试对手的移动
            test_state[move] = -player
            result = self.end_game(test_state)
            test_state[move] = 0
            
            if result is not None:
                winner = result.argmax()
                if (winner == 0 and -player == BLACK) or (winner == 2 and -player == WHITE):
                    # 检查玩家是否在此位置下子
                    if state[move] == player:
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
        if not moves:  # 如果没有可用的移动
            return None
            
        if (random.random() < self.epsilon) or (state_key not in self.Q):
            # 随机探索
            move = random.choice(moves)
        else:
            # 选择Q值最大的动作
            q_values = self.Q[state_key]
            valid_q = [q_values[move[0]*3 + move[1]] for move in moves]
            # 如果所有的Q值都相等或接近相等，添加一些随机性
            if max(valid_q) - min(valid_q) < 0.1:
                # 使用softmax选择动作
                temp = 0.1  # 温度参数
                softmax_probs = np.exp(np.array(valid_q) / temp)
                softmax_probs = softmax_probs / np.sum(softmax_probs)
                move_idx = np.random.choice(len(moves), p=softmax_probs)
                move = moves[move_idx]
            else:
                move = moves[np.argmax(valid_q)]
        return move

    def act(self, state, turn):
        """对弈时的行动（全贪婪）"""
        state_key = self.hash(state)
        moves = self.available_moves(state)
        if not moves:  # 如果没有可用的移动
            return None, 0.0
            
        if state_key not in self.Q:
            # 未见过的状态，先检查是否有必胜/必败移动
            for move in moves:
                next_state = state.copy()
                next_state[move] = turn
                if self.end_game(next_state) is not None:
                    result = self.end_game(next_state)
                    winner = result.argmax()
                    if (winner == 0 and turn == BLACK) or (winner == 2 and turn == WHITE):
                        # 必胜移动
                        return move, 1.0
            # 没有必胜移动，优先中心，然后角落
            if (1, 1) in moves:
                return (1, 1), 0.5
            corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
            available_corners = [c for c in corners if c in moves]
            if available_corners:
                return random.choice(available_corners), 0.3
            return random.choice(moves), 0.0
            
        q_values = self.Q[state_key]
        valid_q = [q_values[move[0]*3 + move[1]] for move in moves]
        
        # 增加最优动作选择的鲁棒性
        best_q = max(valid_q)
        best_moves = [moves[i] for i in range(len(moves)) if valid_q[i] >= best_q - 0.001]
        
        if len(best_moves) > 1:
            # 多个最佳动作，考虑额外策略
            # 优先选择能立即获胜的动作
            for move in best_moves:
                next_state = state.copy()
                next_state[move] = turn
                result = self.end_game(next_state)
                if result is not None:
                    winner = result.argmax()
                    if (winner == 0 and turn == BLACK) or (winner == 2 and turn == WHITE):
                        return move, best_q
            # 优先选择中心
            if (1, 1) in best_moves:
                return (1, 1), best_q
            # 优先选择角落
            corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
            corner_moves = [m for m in best_moves if m in corners]
            if corner_moves:
                return random.choice(corner_moves), best_q
        
        return best_moves[0], best_q

    def update_win_lose_states(self, chain, winner):
        """更新胜负状态记录"""
        # 记录最后一个移动作为必胜/必败状态
        if len(chain) >= 2:
            if winner == BLACK:  # 黑胜
                self.win_states[chain[-2]] = True
                self.lose_states[chain[-1]] = True
            elif winner == WHITE:  # 白胜
                self.lose_states[chain[-2]] = True
                self.win_states[chain[-1]] = True

    def train(self, callback=None):
        """增强版训练模型使用Q-Learning方法"""
        # 1. 经验回放缓冲区
        replay_buffer = []
        max_buffer_size = 10000
        
        # 2. 初始化学习率调度
        initial_alpha = self.alpha
        
        for i in tqdm(range(self.count)):
            state = np.zeros((3, 3), dtype=np.int8)
            turn = BLACK
            
            # 更新探索率（随训练进程降低）
            progress = i / self.count
            self.epsilon = max(self.epsilon_min, 
                              self.initial_epsilon * (1 - 0.8 * progress))
            
            # 更新学习率（随训练进程调整）
            self.alpha = initial_alpha * (1 - 0.5 * progress)
            
            # 每100步更新一次训练窗口信息
            if callback and i % 100 == 0:
                callback(i, self.Q, self.epsilon)
            
            # 记录这局游戏中经历的状态序列
            chain = []
            
            # 游戏循环
            while True:
                state_key = self.hash(state)
                chain.append(state_key)
                
                if state_key not in self.Q:
                    self.Q[state_key] = np.zeros(9, dtype=np.float32)
                    self.V[state_key] = 0.0
                
                # 选择动作
                action = self.select_action(state, turn)
                if action is None:  # 没有可用的移动
                    break
                    
                action_idx = action[0] * 3 + action[1]
                
                # 执行动作
                next_state = state.copy()
                next_state[action] = turn
                
                # 获取中间奖励
                intermediate_reward = self.get_reward_shaping(next_state, turn)
                
                # 检查游戏是否结束
                result = self.end_game(next_state)
                if result is not None:
                    # 游戏结束时的奖励
                    winner = None
                    if result[0] == 1:  # 黑胜
                        reward = 1.0 if turn == BLACK else -1.0
                        self.stats["black_wins"] += 1
                        winner = BLACK
                    elif result[2] == 1:  # 白胜
                        reward = 1.0 if turn == WHITE else -1.0
                        self.stats["white_wins"] += 1
                        winner = WHITE
                    else:  # 平局
                        reward = 0.1  # 平局也给予少量正奖励，鼓励不输
                        self.stats["draws"] += 1
                        
                    # 更新胜负状态记录
                    if winner:
                        self.update_win_lose_states(chain, winner)
                    
                    # 更新Q值和状态价值
                    self.Q[state_key][action_idx] += self.alpha * (
                        reward - self.Q[state_key][action_idx])
                    self.V[state_key] = np.max(self.Q[state_key])
                    
                    # 将这局游戏加入经验回放缓冲区
                    replay_buffer.append((state_key, action_idx, reward, None, True))
                    
                    # 控制缓冲区大小
                    if len(replay_buffer) > max_buffer_size:
                        replay_buffer.pop(0)
                    
                    break
                
                # 游戏继续
                next_key = self.hash(next_state)
                if next_key not in self.Q:
                    self.Q[next_key] = np.zeros(9, dtype=np.float32)
                    self.V[next_key] = 0.0
                
                # 将当前步骤加入经验回放缓冲区
                replay_buffer.append((state_key, action_idx, intermediate_reward, next_key, False))
                
                # 控制缓冲区大小
                if len(replay_buffer) > max_buffer_size:
                    replay_buffer.pop(0)
                
                # Q-Learning核心：用"下一个状态的最大Q值"来更新
                next_q = np.max(self.Q[next_key])
                self.Q[state_key][action_idx] += self.alpha * (
                    intermediate_reward + self.gamma * next_q - self.Q[state_key][action_idx])
                
                # 更新状态价值
                self.V[state_key] = np.max(self.Q[state_key])
                
                # 状态转移
                state = next_state
                turn *= -1  # 交换回合
                
            # 经验回放：从缓冲区随机抽取一批经验进行额外学习
            if i % 10 == 0 and replay_buffer:  # 每10局游戏进行一次经验回放
                batch_size = min(32, len(replay_buffer))
                batch = random.sample(replay_buffer, batch_size)
                
                for s_key, a_idx, r, next_key, done in batch:
                    if done:
                        target = r
                    else:
                        target = r + self.gamma * np.max(self.Q[next_key])
                    
                    self.Q[s_key][a_idx] += self.alpha * 0.5 * (target - self.Q[s_key][a_idx])
                    self.V[s_key] = np.max(self.Q[s_key])
                
            # 每1000局保存一次模型
            if i > 0 and i % 1000 == 0:
                self.save()
                
            # 定期显示训练统计
            if i > 0 and i % 5000 == 0:
                total = self.stats["black_wins"] + self.stats["white_wins"] + self.stats["draws"]
                if total > 0:
                    logger.info(f"训练进度 {i}/{self.count} - 状态数: {len(self.Q)} - "
                             f"黑胜率: {self.stats['black_wins']/total:.2f} - "
                             f"白胜率: {self.stats['white_wins']/total:.2f} - "
                             f"平局率: {self.stats['draws']/total:.2f}")

        # 最终保存
        self.save()
        
        # 训练结束时的最后一次回调
        if callback:
            callback(self.count, self.Q, self.epsilon)
            
        # 显示最终训练统计
        total = self.stats["black_wins"] + self.stats["white_wins"] + self.stats["draws"]
        if total > 0:
            logger.info(f"训练完成 - 状态数: {len(self.Q)} - "
                      f"黑胜率: {self.stats['black_wins']/total:.2f} - "
                      f"白胜率: {self.stats['white_wins']/total:.2f} - "
                      f"平局率: {self.stats['draws']/total:.2f}")
