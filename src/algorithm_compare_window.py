import sys
import numpy as np
from PySide6 import QtWidgets, QtCore
from model_qlearning import Model as QLearningModel
from model_sarsa import Model as SarsaModel
from model_mcts import Model as MCTSModel
from model_default import Model as DefaultModel
from game import Game

class AlgorithmCompareWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("算法对比研究")
        self.resize(1200, 700)
        self.init_ui()

    def init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)

        # 参数设置区
        settings_group = QtWidgets.QGroupBox("参数设置区")
        settings_layout = QtWidgets.QFormLayout()
        self.algorithms = ["mcts", "qlearning", "sarsa"]
        self.algorithm_checkboxes = {}
        for algo in self.algorithms:
            cb = QtWidgets.QCheckBox(algo)
            cb.setChecked(True)
            self.algorithm_checkboxes[algo] = cb
            settings_layout.addRow(cb)
        self.epoch_spinbox = QtWidgets.QSpinBox()
        self.epoch_spinbox.setRange(1000, 1000000)
        self.epoch_spinbox.setSingleStep(1000)
        self.epoch_spinbox.setValue(100000)
        settings_layout.addRow("训练轮数:", self.epoch_spinbox)
        self.battle_spinbox = QtWidgets.QSpinBox()
        self.battle_spinbox.setRange(100, 10000)
        self.battle_spinbox.setSingleStep(100)
        self.battle_spinbox.setValue(1000)
        settings_layout.addRow("对战轮数:", self.battle_spinbox)
        settings_group.setLayout(settings_layout)

        # 添加"开始对比"按钮
        self.start_button = QtWidgets.QPushButton("开始对比")
        self.start_button.clicked.connect(self.start_compare)
        settings_layout.addRow(self.start_button)

        # 结果展示区
        result_group = QtWidgets.QGroupBox("结果展示区")
        result_layout = QtWidgets.QGridLayout()
        self.winrate_table = QtWidgets.QTableWidget(3, 4)
        self.winrate_table.setHorizontalHeaderLabels(["模型", "胜率", "平局率", "败率"])
        result_layout.addWidget(QtWidgets.QLabel("胜率对比"), 0, 0)
        result_layout.addWidget(self.winrate_table, 1, 0)
        self.converge_table = QtWidgets.QTableWidget(3, 4)
        self.converge_table.setHorizontalHeaderLabels(["模型", "Q表大小", "Q值均值", "Q值标准差"])
        result_layout.addWidget(QtWidgets.QLabel("收敛速度"), 0, 1)
        result_layout.addWidget(self.converge_table, 1, 1)
        self.strategy_table = QtWidgets.QTableWidget(3, 5)
        self.strategy_table.setHorizontalHeaderLabels(["模型", "中心控制", "角点控制", "边点控制", "其他"])
        result_layout.addWidget(QtWidgets.QLabel("策略分布"), 2, 0)
        result_layout.addWidget(self.strategy_table, 3, 0)
        self.generalize_table = QtWidgets.QTableWidget(3, 3)
        self.generalize_table.setHorizontalHeaderLabels(["模型", "泛化胜率", "泛化对手"])
        result_layout.addWidget(QtWidgets.QLabel("泛化能力"), 2, 1)
        result_layout.addWidget(self.generalize_table, 3, 1)
        result_group.setLayout(result_layout)

        # 现象总结区
        summary_group = QtWidgets.QGroupBox("现象总结区")
        self.summary_text = QtWidgets.QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout = QtWidgets.QVBoxLayout()
        summary_layout.addWidget(self.summary_text)
        summary_group.setLayout(summary_layout)

        # 主布局
        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addWidget(settings_group, 1)
        top_layout.addWidget(result_group, 3)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(summary_group, 1)

    def start_compare(self):
        # 1. 只加载模型，不训练
        algo_map = {
            "qlearning": QLearningModel,
            "sarsa": SarsaModel,
            "mcts": MCTSModel
        }
        selected = [k for k, v in self.algorithm_checkboxes.items() if v.isChecked()]
        models = {}
        for name in selected:
            model = algo_map[name]()
            try:
                q = model.load()
                # 检查Q表或table是否为空
                if (hasattr(model, 'Q') and not model.Q) or (hasattr(model, 'table') and not model.table):
                    raise Exception('模型未训练')
            except Exception:
                QtWidgets.QMessageBox.warning(self, "模型未训练", f"模型{name}未训练或模型文件不存在，请先在主界面训练后再进行对比！")
                return
            models[name] = model

        # 2. 互相对战
        battle_count = self.battle_spinbox.value()
        results = {name: {"win": 0, "draw": 0, "lose": 0} for name in selected}
        qinfo = {}
        strategy = {}
        for i, name1 in enumerate(selected):
            for j, name2 in enumerate(selected):
                if i == j:
                    continue
                model1 = models[name1]
                model2 = models[name2]
                for _ in range(battle_count):
                    game = Game()
                    game.switch_model(name1)
                    model1 = game.model
                    game2 = Game()
                    game2.switch_model(name2)
                    model2 = game2.model
                    state = np.zeros((3, 3), dtype=np.int8)
                    turn = 1
                    while True:
                        if turn == 1:
                            action, _ = model1.act(state, turn)
                        else:
                            action, _ = model2.act(state, turn)
                        state[action] = turn
                        result = model1.end_game(state)
                        if result is not None:
                            if result[0]:
                                results[name1]["win"] += 1
                                results[name2]["lose"] += 1
                            elif result[1]:
                                results[name1]["draw"] += 1
                                results[name2]["draw"] += 1
                            else:
                                results[name1]["lose"] += 1
                                results[name2]["win"] += 1
                            break
                        turn *= -1
        # 3. Q表信息
        for name in selected:
            model = models[name]
            if hasattr(model, "Q"):
                qtable = model.Q
                qvals = []
                for v in qtable.values():
                    qvals.extend(list(v))
                qinfo[name] = {
                    "size": len(qtable),
                    "mean": float(np.mean(qvals)) if qvals else 0,
                    "std": float(np.std(qvals)) if qvals else 0
                }
            elif hasattr(model, "table"):
                qtable = model.table
                qvals = []
                for v in qtable.values():
                    qvals.extend(list(v))
                qinfo[name] = {
                    "size": len(qtable),
                    "mean": float(np.mean(qvals)) if qvals else 0,
                    "std": float(np.std(qvals)) if qvals else 0
                }
        # 4. 策略分布
        for name in selected:
            model = models[name]
            center, corner, edge = 0, 0, 0
            for _ in range(100):
                state = np.zeros((3, 3), dtype=np.int8)
                action, _ = model.act(state, 1)
                if action == (1, 1):
                    center += 1
                elif action in [(0,0), (0,2), (2,0), (2,2)]:
                    corner += 1
                else:
                    edge += 1
            strategy[name] = {"center": center, "corner": corner, "edge": edge}
        # 5. 泛化能力测试（与其它模型互相比）
        generalize = {}
        for i, name1 in enumerate(selected):
            generalize[name1] = []
            for j, name2 in enumerate(selected):
                if i == j:
                    continue
                model1 = models[name1]
                model2 = models[name2]
                win, draw, lose = 0, 0, 0
                for _ in range(100):
                    state = np.zeros((3, 3), dtype=np.int8)
                    turn = 1
                    while True:
                        if turn == 1:
                            action, _ = model1.act(state, turn)
                        else:
                            action, _ = model2.act(state, turn)
                        state[action] = turn
                        result = model1.end_game(state)
                        if result is not None:
                            if result[0]:
                                win += 1
                            elif result[1]:
                                draw += 1
                            else:
                                lose += 1
                            break
                        turn *= -1
                total = win + draw + lose
                rate = win / total if total else 0
                generalize[name1].append((name2, rate))
        # 填充表格
        self.winrate_table.setRowCount(len(selected))
        for i, name in enumerate(selected):
            total = results[name]["win"] + results[name]["draw"] + results[name]["lose"]
            win_rate = results[name]["win"] / total if total else 0
            draw_rate = results[name]["draw"] / total if total else 0
            lose_rate = results[name]["lose"] / total if total else 0
            self.winrate_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.winrate_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{win_rate:.2%}"))
            self.winrate_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{draw_rate:.2%}"))
            self.winrate_table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{lose_rate:.2%}"))
        self.converge_table.setRowCount(len(selected))
        for i, name in enumerate(selected):
            self.converge_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.converge_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(qinfo[name]["size"])))
            self.converge_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{qinfo[name]['mean']:.3f}"))
            self.converge_table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{qinfo[name]['std']:.3f}"))
        self.strategy_table.setRowCount(len(selected))
        for i, name in enumerate(selected):
            self.strategy_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.strategy_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(strategy[name]["center"])))
            self.strategy_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(strategy[name]["corner"])))
            self.strategy_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(strategy[name]["edge"])))
            self.strategy_table.setItem(i, 4, QtWidgets.QTableWidgetItem("-"))
        # 填充泛化能力表格
        self.generalize_table.setRowCount(len(selected))
        for i, name in enumerate(selected):
            if generalize[name]:
                # 只显示第一个对手的胜率，剩下的用逗号分隔
                rates = [f"{n}:{r:.2%}" for n, r in generalize[name]]
                self.generalize_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
                self.generalize_table.setItem(i, 1, QtWidgets.QTableWidgetItem(", ".join([x.split(":")[1] for x in rates])))
                self.generalize_table.setItem(i, 2, QtWidgets.QTableWidgetItem(", ".join([x.split(":")[0] for x in rates])))
            else:
                self.generalize_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
                self.generalize_table.setItem(i, 1, QtWidgets.QTableWidgetItem("-"))
                self.generalize_table.setItem(i, 2, QtWidgets.QTableWidgetItem("-"))
        # 6. 现象总结
        summary = ""
        for name in selected:
            summary += f"模型{name}：\n胜率：{results[name]['win']}胜/{results[name]['draw']}平/{results[name]['lose']}负\nQ表大小：{qinfo[name]['size']}，均值：{qinfo[name]['mean']:.3f}，标准差：{qinfo[name]['std']:.3f}\n策略分布：中心{strategy[name]['center']}，角{strategy[name]['corner']}，边{strategy[name]['edge']}\n\n"
        self.summary_text.setText(summary)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = AlgorithmCompareWindow()
    window.show()
    sys.exit(app.exec()) 