import os
import sys
import itertools
import functools
import random

import numpy as np
from tqdm import tqdm

from PySide6 import QtWidgets, QtGui, QtCore

from logger import logger
from game import Game
# from model_mcts import BLACK, WHITE, EMPTY
from model_sarsa import BLACK, WHITE, EMPTY
# from model_qlearning import BLACK, WHITE, EMPTY
dirname = os.path.dirname(os.path.abspath(__file__))


class Board(QtWidgets.QLabel):

    action_signal = QtCore.Signal(object)
    refresh_signal = QtCore.Signal(object)

    def __init__(self, parent):
        super().__init__(parent)
        self.piece_size = 200
        self.state = None
        self.last = None

        self.setPixmap(QtGui.QPixmap(os.path.join(dirname, 'images/board.png')))
        self.setScaledContents(True)
        self.resize(self.piece_size * 3, self.piece_size * 3)

        self.pieces: dict[tuple[int, int], QtWidgets.QLabel] = {}

        self.pixmaps = {
            1: QtGui.QPixmap(os.path.join(dirname, 'images/black.png')),
            -1: QtGui.QPixmap(os.path.join(dirname, 'images/white.png'))
        }

        self.border_pixmaps = {
            1: QtGui.QPixmap(os.path.join(dirname, 'images/black_border.png')),
            -1: QtGui.QPixmap(os.path.join(dirname, 'images/white_border.png'))
        }

    def refresh(self, state: np.ndarray, last: tuple[int, int]):
        assert (tuple(state.shape) == (3, 3))
        # logger.debug('refresh %s, %s', state, last)
        self.state = state
        self.last = last

        for where in itertools.product(range(3), range(3)):
            if where not in self.pieces:
                self.pieces[where] = QtWidgets.QLabel(self)
                self.pieces[where].setScaledContents(True)

            piece = self.pieces[where]
            if state[where] == EMPTY:
                piece.setVisible(False)
                continue
            if last == where:
                img = self.border_pixmaps[state[where]]
            else:
                img = self.pixmaps[state[where]]

            piece.setPixmap(img)
            piece.setVisible(True)
            piece.setGeometry(self.pieceGeometry(where))

        super().update()

    def pieceGeometry(self, where: tuple[int, int]):
        return QtCore.QRect(
            where[1] * self.piece_size,
            where[0] * self.piece_size,
            self.piece_size,
            self.piece_size
        )

    def position(self, event: QtGui.QMouseEvent):
        x = event.position().x() // self.piece_size
        y = event.position().y() // self.piece_size

        if x < 0 or x >= 3:
            return None
        if y < 0 or y >= 3:
            return None
        return (int(y), int(x))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.buttons() != QtCore.Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        where = self.position(event)
        logger.debug('click on %s', where)
        self.action_signal.emit(where)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        w = self.parentWidget().width()
        h = self.parentWidget().height() - 30

        width, height = h, h
        if width > w:
            width, height = w, w

        x = (w - width) // 2
        y = (h - height) // 2
        self.setGeometry(x, y, width, height)
        self.piece_size = width // 3

        if self.state is not None:
            self.refresh(self.state, self.last)

        return super().resizeEvent(event)


class ContextMenu(QtWidgets.QMenu):

    reset_signal = QtCore.Signal(None)
    hint_signal = QtCore.Signal(None)
    switch_signal = QtCore.Signal(None)
    undo_signal = QtCore.Signal(None)
    train_signal = QtCore.Signal(None)
    save_signal = QtCore.Signal(None)

    MENUS = [
        ('Restart', 'Ctrl+N', lambda self: self.reset_signal.emit()),
        ('Hint', 'Ctrl+H', lambda self: self.hint_signal.emit()),
        ('Switch', 'Ctrl+K', lambda self: self.switch_signal.emit()),
        ('Undo', 'Ctrl+Z', lambda self: self.undo_signal.emit()),
        ('separator', None, None),
        ('Train', 'Ctrl+T', lambda self: self.train_signal.emit()),
        ('Save', 'Ctrl+S', lambda self: self.save_signal.emit()),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)

        for name, key, slot in self.MENUS:
            if name == 'separator':
                self.addSeparator()
                continue

            if key:
                shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self.parentWidget())
                shortcut.activated.connect(functools.partial(slot, self))

            action = QtGui.QAction(name, self)
            action.setShortcut(key)
            action.triggered.connect(functools.partial(slot, self))
            self.addAction(action)


class TrainingWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("训练状态")
        self.resize(800, 400)  # 调整窗口大小以容纳棋盘
        
        # 使用水平布局
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        
        # 左侧信息区域
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout()
        left_widget.setLayout(left_layout)
        layout.addWidget(left_widget)
        
        # Q表信息
        self.q_info = QtWidgets.QTextEdit()
        self.q_info.setReadOnly(True)
        left_layout.addWidget(QtWidgets.QLabel("Q表状态:"))
        left_layout.addWidget(self.q_info)
        
        # 探索策略信息
        self.strategy_info = QtWidgets.QLabel()
        left_layout.addWidget(QtWidgets.QLabel("探索策略:"))
        left_layout.addWidget(self.strategy_info)
        
        # 训练进度
        self.progress = QtWidgets.QProgressBar()
        left_layout.addWidget(self.progress)
        
        # 右侧棋盘
        self.board = Board(self)
        self.board.setFixedSize(400, 400)  # 固定棋盘大小
        layout.addWidget(self.board)

    def update_info(self, q_table, epsilon, step, total):
        # 更新Q表信息
        q_info_text = f"Q表大小: {len(q_table)}个状态\n"
        
        # 显示部分Q值示例
        if q_table:
            # 找到一个有趣的状态（已经有一些棋子的状态）
            non_empty_states = []
            for state in q_table.keys():
                if isinstance(state, tuple) and len(state) == 9:
                    # 计算状态中非空位置的数量
                    non_empty_count = sum(1 for x in state if x != 0)
                    if non_empty_count > 0:  # 至少有一个棋子
                        non_empty_states.append((state, non_empty_count))
            
            if non_empty_states:
                # 选择一个随机的非空状态
                sample_state = random.choice(non_empty_states)[0]
            else:
                # 如果没有非空状态，就用第一个合法状态
                for state in q_table.keys():
                    if isinstance(state, tuple) and len(state) == 9:
                        sample_state = state
                        break
            
            if sample_state:
                # 将状态转换为3x3棋盘格式
                board = np.array(sample_state, dtype=np.int8).reshape(3, 3)
                
                # 更新棋盘显示
                self.board.refresh(board, None)
                
                # 显示Q值信息
                q_info_text += "\n各位置的Q值:\n"
                q_values = q_table[sample_state]
                q_board = q_values.reshape(3, 3)
                for row in q_board:
                    q_info_text += " ".join(f"{x:6.3f}" for x in row) + "\n"
            
        self.q_info.setText(q_info_text)
        
        # 更新探索策略信息
        strategy_text = f"当前探索率(ε): {epsilon:.2f}"
        self.strategy_info.setText(strategy_text)
        
        # 更新进度
        self.progress.setMaximum(total)
        self.progress.setValue(step)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Tic-Tac-Toe")
        self.setWindowIcon(QtGui.QIcon(os.path.join(dirname, 'images/black.png')))
        self.statusBar().showMessage("Tic-Tac-Toe")

        self.board = Board(self)
        self.resize(self.board.size())
        self.board.action_signal.connect(self.action)

        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setMaximumWidth(500)
        self.statusBar().addPermanentWidget(self.progressBar)
        self.progressBar.setVisible(False)

        self.messageBox = QtWidgets.QMessageBox(self)
        self.contextMenu = ContextMenu(self)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)

        self.contextMenu.reset_signal.connect(self.restart)
        self.contextMenu.hint_signal.connect(self.hint)
        self.contextMenu.switch_signal.connect(self.switch)
        self.contextMenu.undo_signal.connect(self.undo)
        self.contextMenu.train_signal.connect(self.train)
        self.contextMenu.save_signal.connect(self.save)

        self.game = Game(WHITE)

        self.restart()

        self.training_window = TrainingWindow(self)
        
    def showContextMenu(self, point):
        self.contextMenu.exec(self.mapToGlobal(point))

    def restart(self):
        logger.info("Restart Game")
        self.statusBar().showMessage("Restart Game...")
        self.game.reset()
        self.board.refresh(self.game.state, None)
        self.ai_action()

    def hint(self):
        where, confidence = self.game.model.act(self.game.state, self.game.turn)
        logger.debug(f"model turn {self.game.turn} act {where} confidence {confidence: 0.2f}")
        self.action(where, confidence)

    def undo(self):
        self.game.undo()
        self.board.refresh(self.game.state, self.game.last)

    def save(self):
        logger.info('save mode to %s', self.game.model.filename)
        self.game.model.save()

    def switch(self):
        self.game.ai *= -1
        self.ai_action()

    def ai_action(self):
        if self.game.turn != self.game.ai:
            return
        self.hint()
        if self.check():
            return

    def check(self):
        r = self.game.model.end_game(self.game.state)
        if r is None:
            return False

        black, draw, white = r
        message = None
        if black:
            message = 'Black win'
            image = os.path.join(dirname, 'images/black.png')
        if white:
            message = 'White win'
            image = os.path.join(dirname, 'images/white.png')
        if draw:
            message = 'Draw'
            image = os.path.join(dirname, 'images/draw.png')

        self.statusBar().showMessage(message)
        self.messageBox.setWindowIcon(QtGui.QIcon(image))
        self.messageBox.setIconPixmap(QtGui.QPixmap(image).scaledToWidth(200))
        self.messageBox.setWindowTitle("Inform")
        self.messageBox.setText(message)
        self.messageBox.show()
        # self.messageBox.exec()

        return True

    def action(self, where: tuple[int, int], confidence: float = 0.0):
        self.statusBar().showMessage(f"Action {where} confidence {confidence:0.2f}...")
        if self.game.state[where] != EMPTY:
            return
        if self.check():
            return

        self.game.action(where)
        self.board.refresh(self.game.state, where)
        if self.check():
            return

        if self.game.turn == self.game.ai:
            self.hint()

        if self.check():
            return

    def train(self):
        self.training_window.show()
        self.progressBar.setVisible(True)
        self.progressBar.setRange(0, self.game.model.count)

        # 修改model的训练函数，使其能够回调更新信息
        def update_callback(step, q_table, epsilon):
            self.training_window.update_info(q_table, epsilon, step, self.game.model.count)
            QtWidgets.QApplication.processEvents()  # 保持UI响应
            
        self.game.model.train(callback=update_callback)

        self.progressBar.setValue(self.game.model.count)
        self.progressBar.setVisible(False)
        self.save()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self.board.resizeEvent(event)
        return super().resizeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
