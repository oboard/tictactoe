import matplotlib.pyplot as plt
import numpy as np

# Data from screenshot
models = ['mcts', 'qlearning', 'sarsa']
win_rates = [0.5500, 0.3420, 0.5795]
q_table_sizes = [4520, 1091, 4520]
center = [0, 12, 0]
corner = [0, 49, 100]
edge = [100, 39, 0]
generalize = [[0.79, 1.00], [0.50, 0.45], [1.00, 0.85]]
generalize_labels = [['qlearning', 'sarsa'], ['mcts', 'sarsa'], ['mcts', 'qlearning']]

# 1. Win Rate Bar Chart
plt.figure(figsize=(6,4))
plt.bar(models, [w*100 for w in win_rates], color=['#4C72B0', '#55A868', '#C44E52'])
plt.ylabel('Win Rate (%)')
plt.title('Model Win Rate Comparison')
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('win_rate_comparison.png')
plt.show()

# 2. Q-table Size Bar Chart
plt.figure(figsize=(6,4))
plt.bar(models, q_table_sizes, color=['#4C72B0', '#55A868', '#C44E52'])
plt.ylabel('Q-table Size')
plt.title('Q-table Size Comparison')
plt.tight_layout()
plt.savefig('qtable_size_comparison.png')
plt.show()

# 3. Strategy Distribution Stacked Bar Chart
plt.figure(figsize=(7,5))
bar1 = np.array(center)
bar2 = np.array(corner)
bar3 = np.array(edge)
plt.bar(models, bar1, label='Center', color='#4C72B0')
plt.bar(models, bar2, bottom=bar1, label='Corner', color='#55A868')
plt.bar(models, bar3, bottom=bar1+bar2, label='Edge', color='#C44E52')
plt.ylabel('Count (out of 100)')
plt.title('Strategy Distribution (First Move)')
plt.legend()
plt.tight_layout()
plt.savefig('strategy_distribution.png')
plt.show()

# 4. Generalization Ability Grouped Bar Chart
plt.figure(figsize=(8,5))
bar_width = 0.35
x = np.arange(len(models))
for i in range(2):
    plt.bar(x + i*bar_width, [g[i]*100 for g in generalize], width=bar_width, label=f'vs {generalize_labels[0][i]}')
plt.xticks(x + bar_width/2, models)
plt.ylabel('Generalization Win Rate (%)')
plt.title('Generalization Ability (Win Rate vs Other Models)')
plt.ylim(0, 110)
plt.legend()
plt.tight_layout()
plt.savefig('generalization_ability.png')
plt.show() 