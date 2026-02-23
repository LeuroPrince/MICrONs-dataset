print("正在构建 1059 x 1059 内部连接矩阵 W...")
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

##从之前储存在缓存中的数据加载需要的神经元列表与突触数据
CACHE_DIR = './cache'  

# 加载全局神经元 ID 列表
with open(os.path.join(CACHE_DIR, 'global_root_ids.pkl'), 'rb') as f:
    global_root_ids = pickle.load(f)

# 加载输入突触数据
with open(os.path.join(CACHE_DIR, 'synapses_in.pkl'), 'rb') as f:
    synapses_in_df_fast = pickle.load(f)

N_global = len(global_root_ids)
print(f"已加载 {N_global} 个神经元和 {len(synapses_in_df_fast)} 条突触记录。")

# 仅保留发送方 (pre) 也在这 1059 个列表中的突触
recurrent_synapses = synapses_in_df_fast[synapses_in_df_fast['pre_pt_root_id'].isin(global_root_ids)]
print(f"在这 {N_global} 个神经元内部，共找到了 {len(recurrent_synapses)} 条相互连接。")

# 统计两两之间的突触数量
recurrent_counts = recurrent_synapses.groupby(['post_pt_root_id', 'pre_pt_root_id']).size().reset_index(name='weight')

# 数据透视，行=接收方，列=发送方
W_df = recurrent_counts.pivot(index='post_pt_root_id', columns='pre_pt_root_id', values='weight').fillna(0)

# 强制重排，保证 1059 x 1059 的对应
W_df = W_df.reindex(index=global_root_ids, columns=global_root_ids, fill_value=0)
W_global = W_df.values

# 计算稀疏度
density = np.count_nonzero(W_global) / (N_global * N_global) * 100
print(f"-> 突触连接矩阵 W 构建完成，形状为: {W_global.shape}")
print(f"-> 网络的真实连接密度为: {density:.2f}%")


#计算percentile

non_zero_weights = W_global[W_global > 0]

# 计算 99% 分位数作为上限 (可以根据需要调整为 95 或 99.9)
# 这意味着我们会忽略掉最高的 1% 的离群值，让剩下的 99% 的连接显示得更清楚
vmax_percentile = np.percentile(non_zero_weights, 99) if len(non_zero_weights) > 0 else 1

print(f"突触数最大值: {np.max(W_global)}")
print(f"设定的 99% 分位数上限 (vmax): {vmax_percentile:.2f}")
# 绘制突触连接矩阵热图
print("开始绘制全局连接矩阵...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 图1：使用分位数优化的权重热力图
# 我们使用 LogNorm，并将 vmax 设置为刚刚计算出来的分位数
sns.heatmap(W_global,
            cmap='rocket_r',
            square=True,
            ax=axes[0],
            xticklabels=False,
            yticklabels=False,
            # 使用分位数作为上限，vmin 设置为 1 确保只显示有连接的
            norm=LogNorm(vmin=1, vmax=vmax_percentile),
            cbar_kws={'label': f'Synapse Count (capped at 99th percentile: {vmax_percentile:.1f})'})

axes[0].set_title(f"Recurrent Connectivity (W)\n(Normalized by 99th Percentile)")
axes[0].set_xlabel("Post-synaptic (Receiver)")
axes[0].set_ylabel("Pre-synaptic (Sender)")

# 图 2：Binary拓扑图
axes[1].spy(W_global, markersize=0.2, color='black')
axes[1].set_title("Binary Connection Pattern (Topology)")
axes[1].set_xlabel("Post-synaptic (Receiver)")
axes[1].set_ylabel("Pre-synaptic (Sender)")
axes[1].set_aspect('equal')
plt.show()