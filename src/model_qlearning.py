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
        self.initial_epsilon = epsilon  # 初始探索率
        self.epsilon = epsilon          # 当前探索率
        self.epsilon_decay = 0.9995     # 探索率衰减系数
        self.epsilon_min = 0.01         # 最小探索率
        self.alpha = alpha              # 学习率
        self.gamma = gamma              # 折扣因子
        self.count = count              # 训练回合数
        self.filename = os.path.join(dirname, "model_qlearning.pkl")
        self.Q = self.load()            # Q表：key=状态, value=动作Q值数组
        self.V = {}                     # 状态价值表：key=状态, value=状态价值
        logger.info("load Q-learning state count %s", len(self.Q))

    def save(self):
        """保存模型"""
        # 将Q表和V表转换为可序列化的格式
        q_data = {str(k): v.tolist() for k, v in self.Q.items()}
        v_data = {str(k): v for k, v in self.V.items()}
        with open(self.filename, 'wb') as f:
            pickle.dump((q_data, v_data), f)

    def load(self):
        """加载模型"""
        if not os.path.exists(self.filename):
            return {}
        try:
            with open(self.filename, 'rb') as f:
                q_data, v_data = pickle.load(f)
                # 将字符串键转换回元组，并将列表转换回numpy数组
                self.Q = {eval(k): np.array(v, dtype=np.float32) for k, v in q_data.items()}
                self.V = {eval(k): v for k, v in v_data.items()}
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
        # 将元组转换为字符串作为键
        return str(canonical_state)

    def available_moves(self, state):
        return [tuple(x) for x in np.argwhere(state == EMPTY)]

    def get_reward_shaping(self, state: np.ndarray, turn: int) -> float:
        """奖励塑形：根据当前状态给予中间奖励"""
        reward = 0.0
        
        # 检查行、列和对角线
        for i in range(3):
            # 检查行
            row_sum = np.sum(state[i, :])
            if abs(row_sum) == 2:
                reward += 0.1 * np.sign(row_sum) * turn
            # 检查列
            col_sum = np.sum(state[:, i])
            if abs(col_sum) == 2:
                reward += 0.1 * np.sign(col_sum) * turn
                
        # 检查对角线
        diag_sum = np.trace(state)
        if abs(diag_sum) == 2:
            reward += 0.1 * np.sign(diag_sum) * turn
            
        anti_diag_sum = np.trace(np.flip(state, axis=1))
        if abs(anti_diag_sum) == 2:
            reward += 0.1 * np.sign(anti_diag_sum) * turn
            
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
            # 如果所有的Q值都相等，随机选择
            if len(set(valid_q)) == 1:
                move = random.choice(moves)
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
            move = random.choice(moves)
            return move, 0.0
            
        q_values = self.Q[state_key]
        valid_q = [q_values[move[0]*3 + move[1]] for move in moves]
        # 如果所有的Q值都相等，随机选择
        if len(set(valid_q)) == 1:
            move = random.choice(moves)
        else:
            move = moves[np.argmax(valid_q)]
        return move, np.max(valid_q)

    def train(self, callback=None):
        """训练模型使用Q-Learning方法"""
        for i in tqdm(range(self.count)):
            state = np.zeros((3, 3), dtype=np.int8)
            turn = BLACK
            
            # 更新探索率
            self.epsilon = max(self.epsilon_min, 
                             self.initial_epsilon * (self.epsilon_decay ** i))
            
            # 每100步更新一次训练窗口信息
            if callback and i % 100 == 0:
                callback(i, self.Q, self.epsilon)
            
            while True:
                state_key = self.hash(state)
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
                    winner = result.argmax()
                    if winner == 0:  # 黑胜
                        reward = 1.0 if turn == BLACK else -1.0
                    elif winner == 2:  # 白胜
                        reward = 1.0 if turn == WHITE else -1.0
                    else:  # 平局
                        reward = 0.0
                    # 更新Q值和状态价值
                    self.Q[state_key][action_idx] += self.alpha * (
                        reward - self.Q[state_key][action_idx])
                    self.V[state_key] = reward
                    break
                
                # 游戏继续，选择下一个状态的最大Q值
                next_key = self.hash(next_state)
                if next_key not in self.Q:
                    self.Q[next_key] = np.zeros(9, dtype=np.float32)
                    self.V[next_key] = 0.0
                
                # Q-Learning核心：用"下一个状态的最大Q值"来更新
                next_q = np.max(self.Q[next_key])
                self.Q[state_key][action_idx] += self.alpha * (
                    intermediate_reward + self.gamma * next_q - self.Q[state_key][action_idx])
                
                # 更新状态价值
                self.V[state_key] = np.max(self.Q[state_key])
                
                # 状态转移
                state = next_state
                turn *= -1  # 交换回合

        self.save()
        # 训练结束时的最后一次回调
        if callback:
            callback(self.count, self.Q, self.epsilon)
