import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

CACHE_DIR = './cache'  

##初始化CAVEclient
from caveclient import CAVEclient
client = CAVEclient()
# client.auth.get_new_token()
# client.auth.save_token(token= "295c5e656d7ebbe6ba82dd5521dc4a1a")
client = CAVEclient('minnie65_public')
client.materialize.version = 1621
client.materialize.get_tables()
timestamp = client.materialize.get_timestamp(version=client.materialize.version)

##可视化aibs_metamodel_mtypes_v661_v2表，对其内容有直观的理解
df_layer = client.materialize.live_query('aibs_metamodel_mtypes_v661_v2', timestamp=timestamp)
df_layer.head() 

##现在我们要对aibs_metamodel_mtypes_v661_v2和global_root_ids进行交叉查询
##找到aibs_metamodel_mtypes_v661_v2表中与global_root_id对应的细胞类型和层级信息
##在colab上探索时发现一次性查询aibs_metamodel_mtypes_v661_v2表并在本地过滤虽然慢
##但能避免服务器端的过滤问题导致的下载失败。
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

##加载全局神经元 ID 列表
with open(os.path.join('./cache', 'global_root_ids.pkl'), 'rb') as f:
    global_root_ids = pickle.load(f)

print("正在从 AIBS Metamodel 预测表中提取细胞类型和层级信息...")

# 查数据库中经过深度学习分层的神经元数据
# 我们向 aibs_metamodel_mtypes_v661_v2 表发起查询
# 只过滤出我们关心的高质量神经元 (global_root_ids)
mtypes_table_name = 'aibs_metamodel_mtypes_v661_v2'

print(f"正在下载整个表 '{mtypes_table_name}'，然后进行本地过滤...")
start_time_full_fetch = time.time()
mtypes_full_df = client.materialize.live_query(mtypes_table_name, timestamp=timestamp)
end_time_full_fetch = time.time()
print(f"-> 整个表 '{mtypes_table_name}' 下载完成。耗时: {end_time_full_fetch - start_time_full_fetch:.2f} 秒。共 {len(mtypes_full_df)} 条记录。")

# Now, filter locally using the global_root_ids
mtypes_df = mtypes_full_df[mtypes_full_df['pt_root_id'].isin(global_root_ids)].copy()

print(f"-> 本地过滤完成。共找到了 {len(mtypes_df)} 条匹配的记录，用于后续分析。")

# 步骤 2: 将细分的亚型映射为宏观解剖层级 (Broad Layer)
# 根据官方文档说明，L2a-L3c 属于 L2/3；L4a-L4c 属于 L4...
def extract_broad_layer(cell_type):
    if not isinstance(cell_type, str):
        return 'Unknown'

    ct_upper = cell_type.upper()
    if ct_upper.startswith('L2') or ct_upper.startswith('L3'):
        return 'L2/3'
    elif ct_upper.startswith('L4'):
        return 'L4'
    elif ct_upper.startswith('L5'):
        return 'L5'
    elif ct_upper.startswith('L6'):
        return 'L6'
    else:
        return 'Unknown'
    
    # 应用函数到 DataFrame，创建新的 'broad_layer' 列
mtypes_df['broad_layer'] = mtypes_df['cell_type'].apply(extract_broad_layer)

print("已为 mtypes_df 添加 'broad_layer' 列。")
mtypes_df.head()


# 步骤 3: 与全局 ID 对齐并构建排序索引
LINE_POSITIONS_CACHE = os.path.join(CACHE_DIR, 'line_positions.pkl')

if os.path.exists(LINE_POSITIONS_CACHE):
    print("加载缓存的 line_positions 数据...")
    with open(LINE_POSITIONS_CACHE, 'rb') as f:
        line_positions = pickle.load(f)
    # Also need sorted_indices for W_sorted calculation if not already available
    # For simplicity, if line_positions are cached, we assume sorting_df and sorted_indices are either already in memory or will be recreated correctly.
    # However, for a robust solution, you might want to cache sorting_df or relevant parts.
    # For this example, let's ensure sorting_df is always calculated to get sorted_indices.
    print("line_positions 已从缓存加载。")

else:
    print("正在计算 line_positions 数据...")
    # 用 left merge 保证与之前的W_global对齐，且不会丢失任何 global_root_id
    sorting_df = pd.DataFrame({'pt_root_id': global_root_ids})
    sorting_df = sorting_df.merge(mtypes_df[['pt_root_id', 'cell_type', 'broad_layer']], on='pt_root_id', how='left')

    # 对于极少数模型未能成功预测的细胞，我们标记为 Unknown
    sorting_df['broad_layer'] = sorting_df['broad_layer'].fillna('Unknown')
    sorting_df['cell_type'] = sorting_df['cell_type'].fillna('Unknown')

    # 赋予排序权重：L2/3 -> L4 -> L5 -> L6 -> Unknown
    layer_order = {'L2/3': 1, 'L4': 2, 'L5': 3, 'L6': 4, 'Unknown': 5}
    sorting_df['layer_rank'] = sorting_df['broad_layer'].map(layer_order)

    # 双重排序 ：先按层级排，同一层内再按具体的 cell_type (如 L4a, L4b) 内部排序！
    sorting_df = sorting_df.sort_values(by=['layer_rank', 'cell_type'])

    # 提取排序后在原数组中的下标 (Indices)
    sorted_indices = sorting_df.index.tolist()
    sorted_labels = sorting_df['broad_layer'].tolist()

    # 找出层级更替的分界点
    line_positions = np.where(np.array(sorted_labels[:-1]) != np.array(sorted_labels[1:]))[0] + 1
    
    with open(LINE_POSITIONS_CACHE, 'wb') as f:
        pickle.dump(line_positions, f)
    print("line_positions 已计算并保存到缓存。")

#  为了后续W_sorted的计算，我们需要确保 sorted_indices 是可用的。    
sorting_df = pd.DataFrame({'pt_root_id': global_root_ids})
sorting_df = sorting_df.merge(mtypes_df[['pt_root_id', 'cell_type', 'broad_layer']], on='pt_root_id', how='left')
sorting_df['broad_layer'] = sorting_df['broad_layer'].fillna('Unknown')
sorting_df['cell_type'] = sorting_df['cell_type'].fillna('Unknown')
layer_order = {'L2/3': 1, 'L4': 2, 'L5': 3, 'L6': 4, 'Unknown': 5}
sorting_df['layer_rank'] = sorting_df['broad_layer'].map(layer_order)
sorting_df = sorting_df.sort_values(by=['layer_rank', 'cell_type'])
sorted_indices = sorting_df.index.tolist()

# 步骤 4: 绘制连接矩阵 W

##从之前储存在缓存中的数据加载需要的神经元列表与突触数据
# 加载全局神经元 ID 列表
with open(os.path.join(CACHE_DIR, 'global_root_ids.pkl'), 'rb') as f:
    global_root_ids = pickle.load(f)

# 加载输入突触数据
with open(os.path.join(CACHE_DIR, 'synapses_in.pkl'), 'rb') as f:
    synapses_in_df_fast = pickle.load(f)

N_global = len(global_root_ids)
print(f"已加载 {N_global} 个神经元和 {len(synapses_in_df_fast)} 条突触记录。")

# 仅保留发送方 (pre) 也在这 1451 个列表中的突触
recurrent_synapses = synapses_in_df_fast[synapses_in_df_fast['pre_pt_root_id'].isin(global_root_ids)]
print(f"在这 {N_global} 个神经元内部，共找到了 {len(recurrent_synapses)} 条相互连接。")

# 统计两两之间的突触数量
recurrent_counts = recurrent_synapses.groupby(['pre_pt_root_id', 'post_pt_root_id']).size().reset_index(name='weight')

# 数据透视，行=发送方，列=接收方
W_df = recurrent_counts.pivot(index='pre_pt_root_id', columns='post_pt_root_id', values='weight').fillna(0)

# 强制重排，保证 1059 x 1059 的对应
W_df = W_df.reindex(index=global_root_ids, columns=global_root_ids, fill_value=0)
W_global = W_df.values

# W_global 是之前计算出来的 1451x1451 Numpy 矩阵
W_sorted = W_global[np.ix_(sorted_indices, sorted_indices)]

np.save(os.path.join(CACHE_DIR, 'W_sorted.npy'), W_sorted)
np.save(os.path.join(CACHE_DIR, 'W_global.npy'), W_global)

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


# 步骤 5: 绘制优化的分层热力图
from matplotlib.colors import LogNorm

print("正在绘制按分层模型分类后的突触连接图谱...")

fig, ax = plt.subplots(figsize=(10, 8))

# 依然采用 99% 分位数来抑制极端高值，让图表更清晰
non_zero_weights = W_sorted[W_sorted > 0]
vmax_percentile = np.percentile(non_zero_weights, 99) if len(non_zero_weights) > 0 else 5

sns.heatmap(W_sorted, 
            cmap='mako', 
            ax=ax, 
            square=True,
            xticklabels=False, 
            yticklabels=False, 
            norm=LogNorm(vmin=1, vmax=vmax_percentile),
            cbar_kws={'label': f'Synapse Count (capped at {vmax_percentile:.1f})'})

# 画出区分 L2/3, L4, L5, L6 的分界线
for pos in line_positions:
    ax.axhline(pos, color='gray', lw=0.5, ls='--')
    ax.axvline(pos, color='gray', lw=0.5, ls='--')

ax.set_title("Recurrent Connectivity (Sorted by Predicted Layer)")
ax.set_xlabel("Post-synaptic (Receiver: L2/3 $\\rightarrow$ L4 $\\rightarrow$ L5 $\\rightarrow$ L6)")
ax.set_ylabel("Pre-synaptic (Sender: L2/3 $\\rightarrow$ L4 $\\rightarrow$ L5 $\\rightarrow$ L6)")

plt.tight_layout()

current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, 'layer_sorted_Raw_synaptic_connectivity.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi设置分辨率，bbox_inches确保完整保存
print(f'图像已保存到: {save_path}')

plt.show()

print("\n=== 该子集中各层级神经元的数量分布 ===")
print(sorting_df['broad_layer'].value_counts().sort_index())


#将排序后的 W_sorted 矩阵保存为带有层级和神经元ID标签的 CSV 文件，方便后续分析和共享

# CACHE_DIR = './cache'
# with open(os.path.join(CACHE_DIR, 'W_sorted.npy'), 'rb') as f:
#     W_sorted = np.load(f)

# # Create labels for rows and columns based on the sorted order and broad layer
# # sorting_df already contains the global_root_ids and their broad_layers in the correct order
# neuron_labels = [f"{row.broad_layer}_{row.pt_root_id}" for index, row in sorting_df.iterrows()]

# # Convert the W_sorted numpy array to a pandas DataFrame
# W_sorted_df = pd.DataFrame(W_sorted, index=neuron_labels, columns=neuron_labels)

# # Save the DataFrame to a CSV file
# output_filename = 'W_sorted_with_layers.csv'
# W_sorted_df.to_csv(output_filename)

# print(f"'W_sorted' 矩阵已成功保存到 '{output_filename}'，并带有层级和神经元ID标签。")

# # Display the first few rows and columns of the DataFrame to verify
# W_sorted_df.head()