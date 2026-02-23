import os
import pickle
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("正在运用 Coupling Alignment (W @ W.T) 提取高阶拓扑结构...")
CACHE_DIR = './cache'
with open(os.path.join(CACHE_DIR, 'W_sorted.npy'), 'rb') as f:
    W_sorted = np.load(f)
with open(os.path.join(CACHE_DIR, 'line_positions.pkl'), 'rb') as f:
    line_positions = pickle.load(f)

# 1. 计算耦合对齐矩阵 (Coupling Alignment: Shared Outputs)
# 对应scfc代码中的 (J @ J.T)

# W_sorted 是你刚才经过层级排序后的 1059x1059 矩阵
Coupling_Alignment = W_sorted @ W_sorted.T

# 迹归一化 (Trace Normalization)
# Trace（迹）指的是矩阵对角线强度和
N = Coupling_Alignment.shape[0]
Coupling_Alignment = Coupling_Alignment / np.trace(Coupling_Alignment) * N


# 2. 计算共同输入矩阵 (Shared Inputs: W.T @ W)
# 这是额外的补充：看两个神经元是否接收来自同一批上游的信号

Shared_Inputs = W_sorted.T @ W_sorted
Shared_Inputs = Shared_Inputs / np.trace(Shared_Inputs) * N


# 3. 绘图：应用 SymLogNorm 进行高级可视化

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Structural Transformations: Bridging Sparse to Dense (Sorted by Layer)', fontsize=16)

# 图1：原始重排后的物理连接 W (使用 SymLogNorm 技巧)
# linthresh=0.1 表示在 0 附近保持线性，避免 log(0) 报错
norm_W = matplotlib.colors.SymLogNorm(vmin=0, vmax=np.max(W_sorted), base=10, linthresh=0.1)
sns.heatmap(W_sorted, cmap="mako", ax=axes[0], 
            xticklabels=False, yticklabels=False, norm=norm_W,
            cbar_kws={'label': 'Synapses (SymLog Scale)', 'shrink': 0.8})
axes[0].set_title('Raw Connectivity (W)')

# 图2： Coupling Alignment (W @ W.T) 共享输出
norm_CA = matplotlib.colors.SymLogNorm(vmin=0, vmax=np.max(Coupling_Alignment), base=10, linthresh=0.01)
sns.heatmap(Coupling_Alignment, cmap="mako", ax=axes[1], 
            xticklabels=False, yticklabels=False, norm=norm_CA,
            cbar_kws={'label': 'Alignment Strength', 'shrink': 0.8})
axes[1].set_title('Coupling Alignment ($W \cdot W^T$)')

# 图3：  绘制共享输入 (W.T @ W)
norm_SI = matplotlib.colors.SymLogNorm(vmin=0, vmax=np.max(Shared_Inputs), base=10, linthresh=0.01)
sns.heatmap(Shared_Inputs, cmap="mako", ax=axes[2], 
            xticklabels=False, yticklabels=False, norm=norm_SI,
            cbar_kws={'label': 'Alignment Strength', 'shrink': 0.8})
axes[2].set_title('Shared Inputs ($W^T \cdot W$)')

# 画分层网格线 （layer2/3 -> layer4 -> layer5 -> layer 6）
for ax in axes:
    for pos in line_positions:
        ax.axhline(pos, color='white', lw=1, ls=':')
        ax.axvline(pos, color='white', lw=1, ls=':')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, 'coupling_alignment_shared_output_and_inputs.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi设置分辨率，bbox_inches确保完整保存
print(f'图像已保存到: {save_path}')

plt.show()