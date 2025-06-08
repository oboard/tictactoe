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
    model_sarsa_signal = QtCore.Signal(None)
    model_qlearning_signal = QtCore.Signal(None)
    model_mcts_signal = QtCore.Signal(None)
    model_default_signal = QtCore.Signal(None)
    experiment_signal = QtCore.Signal(None)

    MENUS = [
        ('Restart', 'Ctrl+N', lambda self: self.reset_signal.emit()),
        ('Hint', 'Ctrl+H', lambda self: self.hint_signal.emit()),
        ('Switch', 'Ctrl+K', lambda self: self.switch_signal.emit()),
        ('Undo', 'Ctrl+Z', lambda self: self.undo_signal.emit()),
        ('separator', None, None),
        ('Train', 'Ctrl+T', lambda self: self.train_signal.emit()),
        ('Save', 'Ctrl+S', lambda self: self.save_signal.emit()),
        ('separator', None, None),
        ('SARSA', 'Ctrl+1', lambda self: self.model_sarsa_signal.emit()),
        ('Q-Learning', 'Ctrl+2', lambda self: self.model_qlearning_signal.emit()),
        ('MCTS', 'Ctrl+3', lambda self: self.model_mcts_signal.emit()),
        ('Default', 'Ctrl+4', lambda self: self.model_default_signal.emit()),
        ('separator', None, None),
        ('Experiment', 'Ctrl+E', lambda self: self.experiment_signal.emit()),
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
        
        # 使用垂直布局作为主布局
        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)
        
        # 添加模型选择工具栏
        toolbar = QtWidgets.QToolBar()
        main_layout.addWidget(toolbar)
        
        # 添加模型选择单选框组
        model_group = QtWidgets.QButtonGroup(self)
        
        sarsa_btn = QtWidgets.QRadioButton("SARSA")
        sarsa_btn.setChecked(True)  # 默认选中
        model_group.addButton(sarsa_btn)
        toolbar.addWidget(sarsa_btn)
        
        qlearning_btn = QtWidgets.QRadioButton("Q-Learning")
        model_group.addButton(qlearning_btn)
        toolbar.addWidget(qlearning_btn)
        
        mcts_btn = QtWidgets.QRadioButton("MCTS")
        model_group.addButton(mcts_btn)
        toolbar.addWidget(mcts_btn)
        
        default_btn = QtWidgets.QRadioButton("Default")
        model_group.addButton(default_btn)
        toolbar.addWidget(default_btn)
        
        # 添加工具栏分隔符
        toolbar.addSeparator()
        
        # 添加训练轮数设置
        epoch_layout = QtWidgets.QHBoxLayout()
        epoch_label = QtWidgets.QLabel("训练轮数:")
        self.epoch_spinbox = QtWidgets.QSpinBox()
        self.epoch_spinbox.setRange(1000, 100000)  # 设置训练轮数范围
        self.epoch_spinbox.setSingleStep(1000)     # 每次调整增减1000轮
        self.epoch_spinbox.setValue(30000)         # 默认30000轮
        epoch_layout.addWidget(epoch_label)
        epoch_layout.addWidget(self.epoch_spinbox)
        epoch_widget = QtWidgets.QWidget()
        epoch_widget.setLayout(epoch_layout)
        toolbar.addWidget(epoch_widget)
        
        # 添加工具栏分隔符
        toolbar.addSeparator()
        
        # 添加训练按钮
        train_btn = QtWidgets.QPushButton("开始训练")
        toolbar.addWidget(train_btn)
        
        # 使用水平布局放置主要内容
        content_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        # 左侧信息区域
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout()
        left_widget.setLayout(left_layout)
        content_layout.addWidget(left_widget)
        
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
        self.board.piece_size = 30  # 调整棋子大小
        self.board.setFixedSize(400, 400)  # 调整棋盘大小为 80*3=240
        content_layout.addWidget(self.board)
        
        # 连接按钮信号
        self.model_buttons = {
            'sarsa': sarsa_btn,
            'qlearning': qlearning_btn,
            'mcts': mcts_btn,
            'default': default_btn
        }

        def switch_model(button):
            model_map = {
                sarsa_btn: 'sarsa',
                qlearning_btn: 'qlearning',
                mcts_btn: 'mcts',
                default_btn: 'default'
            }
            if button.isChecked():
                model_type = model_map[button]
                self.parent().switch_model(model_type)
                # 取消其他按钮的选中状态
                for btn in model_map.keys():
                    if btn != button:
                        btn.setChecked(False)
                
        model_group.buttonClicked.connect(switch_model)
        train_btn.clicked.connect(self.start_training)
        
    def start_training(self):
        # 设置新的训练轮数
        epochs = self.epoch_spinbox.value()
        self.parent().game.model.count = epochs
        # 开始训练
        self.parent().train()

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
                
                # 显示Q值/概率信息
                q_info_text += "\n各位置的值:\n"
                values = q_table[sample_state]
                
                # 根据值的不同格式进行显示
                if isinstance(values, np.ndarray):
                    if values.size == 3:  # Default模型 [黑胜,平,白胜]
                        q_info_text += f"黑胜概率: {values[0]:.3f}\n"
                        q_info_text += f"平局概率: {values[1]:.3f}\n"
                        q_info_text += f"白胜概率: {values[2]:.3f}\n"
                    else:  # SARSA/Q-Learning模型 9个Q值
                        values_board = values.reshape(3, 3)
                        for row in values_board:
                            q_info_text += " ".join(f"{x:6.3f}" for x in row) + "\n"
                else:
                    q_info_text += str(values) + "\n"
            
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

        # 连接所有信号
        self.contextMenu.reset_signal.connect(self.restart)
        self.contextMenu.hint_signal.connect(self.hint)
        self.contextMenu.switch_signal.connect(self.switch)
        self.contextMenu.undo_signal.connect(self.undo)
        self.contextMenu.train_signal.connect(self.train)
        self.contextMenu.save_signal.connect(self.save)
        self.contextMenu.experiment_signal.connect(self.show_experiment)
        
        # 添加模型切换信号连接
        self.contextMenu.model_default_signal.connect(lambda: self.switch_model('default'))
        self.contextMenu.model_sarsa_signal.connect(lambda: self.switch_model('sarsa'))
        self.contextMenu.model_qlearning_signal.connect(lambda: self.switch_model('qlearning'))
        self.contextMenu.model_mcts_signal.connect(lambda: self.switch_model('mcts'))

        self.current_model = 'default'
        self.game = Game(WHITE)

        self.restart()
        self.training_window = TrainingWindow(self)
        self.experiment_window = None  # 初始化实验窗口
        self.algorithm_compare_window = None  # 新增算法对比窗口引用
        
        # 添加算法对比菜单项
        menubar = self.menuBar()
        compare_menu = menubar.addMenu("算法对比")
        compare_action = QtGui.QAction("打开算法对比研究窗口", self)
        compare_action.triggered.connect(self.show_algorithm_compare)
        compare_menu.addAction(compare_action)

    def switch_model(self, model_type):
        """切换AI模型"""
        if model_type == self.current_model:
            return
            
        self.current_model = model_type
            
        self.game.switch_model(model_type)
        
        # 更新训练窗口中的单选框状态
        if hasattr(self, 'training_window'):
            self.training_window.model_buttons[model_type].setChecked(True)
        
        # 更新状态栏
        self.statusBar().showMessage(f"已切换到 {model_type.upper()} 模型")

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
        self.progressBar.setRange(0, self.game.model.count)  # 使用更新后的训练轮数

        # 修改model的训练函数，使其能够回调更新信息
        def update_callback(step, q_table, epsilon):
            self.training_window.update_info(q_table, epsilon, step, self.game.model.count)
            QtWidgets.QApplication.processEvents()  # 保持UI响应
            
        self.game.model.train(callback=update_callback)

        self.progressBar.setValue(self.game.model.count)
        self.progressBar.setVisible(False)
        
        # 根据当前模型类型保存到对应文件
        model_files = {
            'sarsa': 'model_sarsa.pkl',
            'qlearning': 'model_qlearning.pkl',
            'mcts': 'model_mcts.pkl',
            'default': 'model_default.pkl'
        }
        self.game.model.filename = model_files[self.current_model]  # 设置保存文件名
        self.save()

    def show_experiment(self):
        """显示实验窗口"""
        if self.experiment_window is None:
            from experiment_window import ExperimentWindow
            self.experiment_window = ExperimentWindow(self)
        self.experiment_window.show()

    def show_algorithm_compare(self):
        if self.algorithm_compare_window is None:
            from algorithm_compare_window import AlgorithmCompareWindow
            self.algorithm_compare_window = AlgorithmCompareWindow()
        self.algorithm_compare_window.show()

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
