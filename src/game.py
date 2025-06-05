import re
import numpy as np
# from model_mcts import BLACK, EMPTY, WHITE, Model
from model_sarsa import BLACK, EMPTY, WHITE, Model
# from model_qlearning import BLACK, WHITE, EMPTY, Model

class Game(object):
    def __init__(self, ai=WHITE) -> None:
        self.state = np.zeros((3, 3), dtype=np.int8)   # 初始状态，全是 0
        self.turn = BLACK                                    # 黑棋先手
        self.model = Model()
        
        # 尝试加载已训练的模型
        try:
            self.model.load('model_sarsa.pkl')  # 加载已保存的模型
            print("成功加载训练模型: model_sarsa.pkl")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("将使用未训练的新模型")
            
        self.ai = ai
        self.last = None
        self.stack = []

    def reset(self):
        self.last = None
        self.stack = []
        self.state = np.zeros((3, 3), dtype=np.int8)
        self.turn = BLACK

    def input_action(self):
        while True:
            index = input("please input action (1 ~ 9):")
            if not re.match('[1-9]', index):
                continue
            index = int(index) - 1
            where = (index // 3, index % 3)
            if self.state[where] != EMPTY:
                continue
            return where

    def action(self, where: tuple[int, int]):
        assert (self.state[where] == 0)
        self.stack.append((self.state.copy(), self.turn, where))
        self.state[where] = self.turn
        self.last = where
        self.turn *= -1

    def undo(self, count=2):
        if len(self.stack) < count:
            return
        self.state, self.turn, self.last = self.stack[-count]
        self.stack = self.stack[:-count]

    def check(self):
        r = self.model.end_game(self.state)
        if r is None:
            return False
        print(self.state)
        black, draw, white = r
        if black:
            print('black win')
        if white:
            print('white win')
        if draw:
            print('draw')
        return True

    def start(self):
        while True:
            print(self.state)
            if self.ai == self.turn:
                where, confidence = self.model.act(self.state, self.turn)
            else:
                where = self.input_action()
            self.action(where)
            if self.check():
                break


def main():
    game = Game()
    game.ai = BLACK
    game.start()


if __name__ == '__main__':
    main()
