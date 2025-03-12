import matplotlib.pyplot as plt

# 创建图形和轴
fig, ax = plt.subplots(figsize=(12, 6))

# 定义文本框的样式
bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")

# 定义箭头的样式
arrowprops = dict(arrowstyle="->", lw=1.5, color='black')

# 绘制输入框
ax.text(0.5, 0.95, '输入', ha='center', va='center', bbox=bbox_props, fontsize=12)

# 决策树的位置
tree_positions = [0.2, 0.4, 0.6, 0.8]
trees = ['决策树A', '决策树B', '决策树C', '决策树D']

# 绘制决策树
for i, pos in enumerate(tree_positions):
    ax.text(pos, 0.75, trees[i], ha='center', va='center', bbox=bbox_props, fontsize=12)
    ax.annotate("", xy=(pos, 0.87), xytext=(0.5, 0.9), arrowprops=arrowprops)

# 绘制结合器和输出框
ax.text(0.5, 0.5, '结合器', ha='center', va='center', bbox=bbox_props, fontsize=12)
ax.text(0.5, 0.35, '输出', ha='center', va='center', bbox=bbox_props, fontsize=12)

# 绘制决策树的子节点和箭头
for pos in tree_positions:
    # 绘制子节点
    ax.text(pos - 0.05, 0.65, '', ha='center', va='center', bbox=bbox_props, fontsize=12)
    ax.text(pos + 0.05, 0.65, '', ha='center', va='center', bbox=bbox_props, fontsize=12)
    ax.text(pos - 0.1, 0.55, '', ha='center', va='center', bbox=bbox_props, fontsize=12)
    ax.text(pos, 0.55, '', ha='center', va='center', bbox=bbox_props, fontsize=12)
    ax.text(pos + 0.1, 0.55, '', ha='center', va='center', bbox=bbox_props, fontsize=12)

    # 绘制从决策树到子节点的箭头
    ax.annotate("", xy=(pos, 0.75), xytext=(pos - 0.05, 0.65), arrowprops=arrowprops)
    ax.annotate("", xy=(pos, 0.75), xytext=(pos + 0.05, 0.65), arrowprops=arrowprops)
    ax.annotate("", xy=(pos - 0.05, 0.65), xytext=(pos - 0.1, 0.55), arrowprops=arrowprops)
    ax.annotate("", xy=(pos - 0.05, 0.65), xytext=(pos, 0.55), arrowprops=arrowprops)
    ax.annotate("", xy=(pos + 0.05, 0.65), xytext=(pos, 0.55), arrowprops=arrowprops)
    ax.annotate("", xy=(pos + 0.05, 0.65), xytext=(pos + 0.1, 0.55), arrowprops=arrowprops)

    # 绘制从子节点到结合器的箭头
    ax.annotate("", xy=(pos - 0.1, 0.55), xytext=(0.5, 0.5), arrowprops=arrowprops)
    ax.annotate("", xy=(pos, 0.55), xytext=(0.5, 0.5), arrowprops=arrowprops)
    ax.annotate("", xy=(pos + 0.1, 0.55), xytext=(0.5, 0.5), arrowprops=arrowprops)

# 绘制从结合器到输出的箭头
ax.annotate("", xy=(0.5, 0.5), xytext=(0.5, 0.35), arrowprops=arrowprops)

# 隐藏坐标轴
ax.axis('off')

# 显示图
plt.show()
