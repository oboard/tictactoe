import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from PySide6 import QtWidgets, QtCore, QtGui
from tqdm import tqdm

from game import Game
from model_default import Model as DefaultModel
from model_sarsa import Model as SarsaModel
from model_qlearning import Model as QLearningModel
from model_mcts import Model as MCTSModel

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理NumPy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class ExperimentWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI实验分析")
        self.resize(1200, 800)
        
        # 初始化模型字典
        self.models = {}
        
        # 创建主布局
        main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(main_layout)
        
        # 左侧控制面板
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(300)
        main_layout.addWidget(control_panel)
        
        # 实验设置组
        settings_group = QtWidgets.QGroupBox("实验设置")
        settings_layout = QtWidgets.QFormLayout()
        settings_group.setLayout(settings_layout)
        
        # 对战轮数设置
        self.battle_spinbox = QtWidgets.QSpinBox()
        self.battle_spinbox.setRange(10, 1000)
        self.battle_spinbox.setSingleStep(10)
        self.battle_spinbox.setValue(100)
        settings_layout.addRow("对战轮数:", self.battle_spinbox)
        
        # 对手选择
        self.opponent_combo = QtWidgets.QComboBox()
        self.opponent_combo.addItems(["MCTS", "RulePlayer"])
        settings_layout.addRow("对手类型:", self.opponent_combo)
        
        # 实验模型选择
        self.model_group = QtWidgets.QButtonGroup()
        self.model_checkboxes = {}
        for model_name in ["default", "sarsa", "qlearning"]:
            checkbox = QtWidgets.QCheckBox(model_name)
            checkbox.setChecked(True)
            self.model_group.addButton(checkbox)
            self.model_checkboxes[model_name] = checkbox
            settings_layout.addRow(checkbox)
            
            # 预加载模型
            if model_name == "default":
                self.models[model_name] = DefaultModel()
            elif model_name == "sarsa":
                self.models[model_name] = SarsaModel()
            else:  # qlearning
                self.models[model_name] = QLearningModel()
        
        control_layout.addWidget(settings_group)
        
        # 实验控制按钮
        button_layout = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("开始实验")
        self.start_button.clicked.connect(self.start_experiment)
        self.export_button = QtWidgets.QPushButton("导出报告")
        self.export_button.clicked.connect(self.export_report)
        self.export_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.export_button)
        control_layout.addLayout(button_layout)
        
        # 实验进度
        self.progress_bar = QtWidgets.QProgressBar()
        control_layout.addWidget(self.progress_bar)
        
        # 右侧结果显示
        result_panel = QtWidgets.QWidget()
        result_layout = QtWidgets.QVBoxLayout()
        result_panel.setLayout(result_layout)
        main_layout.addWidget(result_panel)
        
        # 创建2x2网格布局来放置四个表格
        grid_layout = QtWidgets.QGridLayout()
        result_layout.addLayout(grid_layout)
        
        # 胜率分析
        win_rate_group = QtWidgets.QGroupBox("胜率分析")
        self.win_rate_layout = QtWidgets.QVBoxLayout()
        win_rate_group.setLayout(self.win_rate_layout)
        grid_layout.addWidget(win_rate_group, 0, 0)
        
        # Q表收敛分析
        convergence_group = QtWidgets.QGroupBox("Q表收敛")
        self.convergence_layout = QtWidgets.QVBoxLayout()
        convergence_group.setLayout(self.convergence_layout)
        grid_layout.addWidget(convergence_group, 0, 1)
        
        # 策略分析
        strategy_group = QtWidgets.QGroupBox("策略分析")
        self.strategy_layout = QtWidgets.QVBoxLayout()
        strategy_group.setLayout(self.strategy_layout)
        grid_layout.addWidget(strategy_group, 1, 0)
        
        # 关键局面分析
        critical_group = QtWidgets.QGroupBox("关键局面")
        self.critical_layout = QtWidgets.QVBoxLayout()
        critical_group.setLayout(self.critical_layout)
        grid_layout.addWidget(critical_group, 1, 1)
        
        # 初始化数据存储
        self.experiment_data = {
            "win_rates": {},
            "convergence": {},
            "strategy_analysis": {},
            "critical_positions": {}
        }
        
    def start_experiment(self):
        """开始实验"""
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # 获取实验设置
        battles = self.battle_spinbox.value()
        opponent_type = self.opponent_combo.currentText()
        
        # 创建对手
        if opponent_type == "MCTS":
            opponent = MCTSModel()
        else:
            opponent = DefaultModel()  # RulePlayer使用Default模型
        
        # 对每个选中的模型进行实验
        total_steps = len(self.model_checkboxes) * battles
        current_step = 0
        
        for model_name, checkbox in self.model_checkboxes.items():
            if not checkbox.isChecked():
                continue
                
            # 使用预加载的模型
            model = self.models[model_name]
            
            # 记录Q表收敛情况
            if hasattr(model, 'Q'):
                self.experiment_data["convergence"][model_name] = {
                    "q_table_size": len(model.Q),
                    "q_values": self._analyze_q_values(model.Q)
                }
            
            # 进行对战测试
            wins = 0
            draws = 0
            losses = 0
            
            for _ in tqdm(range(battles)):
                game = Game(1)  # AI先手
                game.switch_model(model_name)
                while True:
                    # AI行动
                    action, _ = model.act(game.state, game.turn)
                    game.action(action)
                    
                    # 检查游戏是否结束
                    result = model.end_game(game.state)
                    if result is not None:
                        if result[0]:  # 黑胜
                            wins += 1
                        elif result[1]:  # 平局
                            draws += 1
                        else:  # 白胜
                            losses += 1
                        break
                    
                    # 对手行动
                    action, _ = opponent.act(game.state, game.turn)
                    game.action(action)
                    
                    # 检查游戏是否结束
                    result = model.end_game(game.state)
                    if result is not None:
                        if result[0]:  # 黑胜
                            wins += 1
                        elif result[1]:  # 平局
                            draws += 1
                        else:  # 白胜
                            losses += 1
                        break
                
                current_step += 1
                self.progress_bar.setValue(int(current_step * 100 / total_steps))
            
            # 记录胜率
            self.experiment_data["win_rates"][model_name] = {
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_rate": wins / battles,
                "draw_rate": draws / battles,
                "loss_rate": losses / battles
            }
            
            # 分析策略
            self.experiment_data["strategy_analysis"][model_name] = self._analyze_strategy(model)
            
            # 分析关键局面
            self.experiment_data["critical_positions"][model_name] = self._analyze_critical_positions(model)
        
        # 更新显示
        self._update_displays()
        self.start_button.setEnabled(True)
        self.export_button.setEnabled(True)
    
    def _analyze_q_values(self, q_table):
        """分析Q值分布"""
        if not q_table:
            return {}
            
        q_values = []
        for state_q in q_table.values():
            if isinstance(state_q, np.ndarray):
                q_values.extend(state_q)
        
        return {
            "mean": np.mean(q_values),
            "std": np.std(q_values),
            "min": np.min(q_values),
            "max": np.max(q_values)
        }
    
    def _analyze_strategy(self, model):
        """分析AI策略"""
        strategy = {
            "center_control": 0,
            "corner_control": 0,
            "edge_control": 0,
            "blocking_moves": 0,
            "winning_moves": 0
        }
        
        # 测试一些典型局面
        test_states = self._generate_test_states()
        for state in test_states:
            action, _ = model.act(state, 1)
            if action == (1, 1):
                strategy["center_control"] += 1
            elif action in [(0,0), (0,2), (2,0), (2,2)]:
                strategy["corner_control"] += 1
            elif action in [(0,1), (1,0), (1,2), (2,1)]:
                strategy["edge_control"] += 1
            
            # 检查是否是防守或进攻动作
            next_state = state.copy()
            next_state[action] = 1
            if model.end_game(next_state) is not None:
                strategy["winning_moves"] += 1
            
            # 检查对手威胁
            for move in [(i, j) for i in range(3) for j in range(3) if state[i,j] == 0]:
                test_state = state.copy()
                test_state[move] = -1
                if model.end_game(test_state) is not None:
                    if action == move:
                        strategy["blocking_moves"] += 1
                    break
        
        return strategy
    
    def _analyze_critical_positions(self, model):
        """分析关键局面"""
        critical = {
            "fork_positions": [],  # 双威胁局面
            "blocking_positions": [],  # 防守局面
            "winning_positions": []  # 必胜局面
        }
        
        # 生成一些关键局面进行测试
        test_states = self._generate_critical_states()
        for state in test_states:
            action, _ = model.act(state, 1)
            
            # 检查是否是双威胁
            threats = 0
            for move in [(i, j) for i in range(3) for j in range(3) if state[i,j] == 0]:
                test_state = state.copy()
                test_state[move] = 1
                if model.end_game(test_state) is not None:
                    threats += 1
            if threats >= 2:
                critical["fork_positions"].append((state, action))
            
            # 检查是否是防守
            for move in [(i, j) for i in range(3) for j in range(3) if state[i,j] == 0]:
                test_state = state.copy()
                test_state[move] = -1
                if model.end_game(test_state) is not None:
                    if action == move:
                        critical["blocking_positions"].append((state, action))
                    break
            
            # 检查是否是必胜
            next_state = state.copy()
            next_state[action] = 1
            if model.end_game(next_state) is not None:
                critical["winning_positions"].append((state, action))
        
        return critical
    
    def _generate_test_states(self):
        """生成测试局面"""
        states = []
        # 空棋盘
        states.append(np.zeros((3, 3), dtype=np.int8))
        # 中心点被占
        state = np.zeros((3, 3), dtype=np.int8)
        state[1,1] = 1
        states.append(state)
        # 角点被占
        state = np.zeros((3, 3), dtype=np.int8)
        state[0,0] = 1
        states.append(state)
        # 边点被占
        state = np.zeros((3, 3), dtype=np.int8)
        state[0,1] = 1
        states.append(state)
        return states
    
    def _generate_critical_states(self):
        """生成关键局面"""
        states = []
        # 双威胁局面
        state = np.zeros((3, 3), dtype=np.int8)
        state[0,0] = 1
        state[0,2] = 1
        state[2,0] = -1
        states.append(state)
        
        # 防守局面
        state = np.zeros((3, 3), dtype=np.int8)
        state[0,0] = 1
        state[0,1] = -1
        state[0,2] = -1
        states.append(state)
        
        # 必胜局面
        state = np.zeros((3, 3), dtype=np.int8)
        state[0,0] = 1
        state[0,1] = 1
        states.append(state)
        
        return states
    
    def _update_displays(self):
        """更新显示结果"""
        # 更新胜率分析
        self._update_win_rate_display()
        
        # 更新Q表收敛分析
        self._update_convergence_display()
        
        # 更新策略分析
        self._update_strategy_display()
        
        # 更新关键局面分析
        self._update_critical_display()
    
    def _update_win_rate_display(self):
        """更新胜率显示"""
        # 清除旧内容
        for i in reversed(range(self.win_rate_layout.count())): 
            self.win_rate_layout.itemAt(i).widget().setParent(None)
        
        # 创建表格
        table = QtWidgets.QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(["模型", "胜场", "平场", "负场", "胜率", "平局率"])
        
        # 填充数据
        table.setRowCount(len(self.experiment_data["win_rates"]))
        for i, (model_name, data) in enumerate(self.experiment_data["win_rates"].items()):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(model_name))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(data["wins"])))
            table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(data["draws"])))
            table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(data["losses"])))
            table.setItem(i, 4, QtWidgets.QTableWidgetItem(f"{data['win_rate']:.2%}"))
            table.setItem(i, 5, QtWidgets.QTableWidgetItem(f"{data['draw_rate']:.2%}"))
        
        table.resizeColumnsToContents()
        self.win_rate_layout.addWidget(table)
    
    def _update_convergence_display(self):
        """更新Q表收敛显示"""
        # 清除旧内容
        for i in reversed(range(self.convergence_layout.count())): 
            self.convergence_layout.itemAt(i).widget().setParent(None)
        
        # 创建表格
        table = QtWidgets.QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["模型", "Q表大小", "Q值均值", "Q值标准差", "Q值范围"])
        
        # 填充数据
        table.setRowCount(len(self.experiment_data["convergence"]))
        for i, (model_name, data) in enumerate(self.experiment_data["convergence"].items()):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(model_name))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(data["q_table_size"])))
            table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{data['q_values']['mean']:.3f}"))
            table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{data['q_values']['std']:.3f}"))
            table.setItem(i, 4, QtWidgets.QTableWidgetItem(
                f"[{data['q_values']['min']:.3f}, {data['q_values']['max']:.3f}]"))
        
        table.resizeColumnsToContents()
        self.convergence_layout.addWidget(table)
    
    def _update_strategy_display(self):
        """更新策略分析显示"""
        # 清除旧内容
        for i in reversed(range(self.strategy_layout.count())): 
            self.strategy_layout.itemAt(i).widget().setParent(None)
        
        # 创建表格
        table = QtWidgets.QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels([
            "模型", "中心控制", "角点控制", "边点控制", "防守动作", "进攻动作"
        ])
        
        # 填充数据
        table.setRowCount(len(self.experiment_data["strategy_analysis"]))
        for i, (model_name, data) in enumerate(self.experiment_data["strategy_analysis"].items()):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(model_name))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(data["center_control"])))
            table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(data["corner_control"])))
            table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(data["edge_control"])))
            table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(data["blocking_moves"])))
            table.setItem(i, 5, QtWidgets.QTableWidgetItem(str(data["winning_moves"])))
        
        table.resizeColumnsToContents()
        self.strategy_layout.addWidget(table)
    
    def _update_critical_display(self):
        """更新关键局面分析显示"""
        # 清除旧内容
        for i in reversed(range(self.critical_layout.count())): 
            self.critical_layout.itemAt(i).widget().setParent(None)
        
        # 创建表格
        table = QtWidgets.QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["模型", "双威胁局面", "防守局面", "必胜局面"])
        
        # 填充数据
        table.setRowCount(len(self.experiment_data["critical_positions"]))
        for i, (model_name, data) in enumerate(self.experiment_data["critical_positions"].items()):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(model_name))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(len(data["fork_positions"]))))
            table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(len(data["blocking_positions"]))))
            table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(len(data["winning_positions"]))))
        
        table.resizeColumnsToContents()
        self.critical_layout.addWidget(table)
    
    def export_report(self):
        """导出实验报告"""
        # 创建报告目录
        report_dir = "experiment_reports"
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        # 生成报告文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(report_dir, f"experiment_report_{timestamp}.json")
        
        # 导出数据
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        
        # 导出Excel报告
        excel_file = os.path.join(report_dir, f"experiment_report_{timestamp}.xlsx")
        with pd.ExcelWriter(excel_file) as writer:
            # 胜率数据
            win_rates_df = pd.DataFrame(self.experiment_data["win_rates"]).T
            win_rates_df.to_excel(writer, sheet_name="胜率分析")
            
            # Q表收敛数据
            convergence_df = pd.DataFrame(self.experiment_data["convergence"]).T
            convergence_df.to_excel(writer, sheet_name="Q表收敛")
            
            # 策略分析数据
            strategy_df = pd.DataFrame(self.experiment_data["strategy_analysis"]).T
            strategy_df.to_excel(writer, sheet_name="策略分析")
            
            # 关键局面数据
            critical_df = pd.DataFrame({
                model: {
                    "双威胁局面": len(data["fork_positions"]),
                    "防守局面": len(data["blocking_positions"]),
                    "必胜局面": len(data["winning_positions"])
                }
                for model, data in self.experiment_data["critical_positions"].items()
            }).T
            critical_df.to_excel(writer, sheet_name="关键局面")
        
        QtWidgets.QMessageBox.information(
            self,
            "导出成功",
            f"实验报告已导出到：\n{report_file}\n{excel_file}"
        ) 