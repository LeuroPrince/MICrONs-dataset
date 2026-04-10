import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from caveclient import CAVEclient

#创立缓存文件夹与缓存文件路径
CACHE_DIR = './cache'                 
os.makedirs(CACHE_DIR, exist_ok=True)

PROOF_CACHE = os.path.join(CACHE_DIR, 'df_proofread_clean.pkl')
COREG_CACHE = os.path.join(CACHE_DIR, 'df_coreg_filtered.pkl')
SYN_IN_CACHE = os.path.join(CACHE_DIR, 'synapses_in.pkl')
SYN_OUT_CACHE = os.path.join(CACHE_DIR, 'synapses_out.pkl')
GLOBAL_IDS_CACHE = os.path.join(CACHE_DIR, 'global_root_ids.pkl')
client = CAVEclient()
# client.auth.get_new_token()
# client.auth.save_token(token= "295c5e656d7ebbe6ba82dd5521dc4a1a")
client = CAVEclient('minnie65_public')
client.materialize.version = 1621
client.materialize.get_tables()
timestamp = client.materialize.get_timestamp(version=client.materialize.version)
if os.path.exists(PROOF_CACHE):
    print("加载缓存的形态学校对数据...")
    with open(PROOF_CACHE, 'rb') as f:
        df_proofread_clean = pickle.load(f)
else:
    print("正在从数据库获取形态学校对数据...")

    table_name_proof = 'proofreading_status_and_strategy'
    df_proofreading = client.materialize.live_query(table_name_proof, timestamp=timestamp)


##清洗数据
# 放宽的校对标准 (OR 逻辑)
    valid_dendrite_strategies = ['dendrite_clean', 'dendrite_extended']
    valid_axon_strategies = ['axon_fully_extended']

    # 提取满足任一条件的神经元
    mask_dendrite_ok = df_proofreading['strategy_dendrite'].isin(valid_dendrite_strategies)
    mask_axon_ok = df_proofreading['strategy_axon'].isin(valid_axon_strategies)

    df_proofread_clean = df_proofreading[mask_dendrite_ok | mask_axon_ok]
    # 保存到缓存
    with open(PROOF_CACHE, 'wb') as f:
        pickle.dump(df_proofread_clean, f)
    print(f"形态学校对数据已缓存，共 {len(df_proofread_clean)} 行。")

# 提取 18位的 神经元 root ID
high_quality_root_ids = df_proofread_clean['pt_root_id'].dropna().drop_duplicates().tolist()
print(f"-> 形态学校对筛选后，保留了 {len(high_quality_root_ids)} 个高质量神经元 (pt_root_id)。")

# 步骤 2: 获取功能配准表并严格按 pt_root_id 取交集
if os.path.exists(COREG_CACHE):
    print("加载缓存的功能配准数据...")
    with open(COREG_CACHE, 'rb') as f:
        df_coreg_filtered = pickle.load(f)
else:
    print("正在从数据库获取功能配准数据...")
    table_name_coreg = 'coregistration_manual_v4'
    print("\n正在获取功能配准表...")
    df_coregistration = client.materialize.live_query(table_name_coreg, timestamp=timestamp)

# 必须严格使用 pt_root_id 进行筛选！
    df_coreg_filtered = df_coregistration[df_coregistration['pt_root_id'].isin(high_quality_root_ids)]
    print(f" 最终结果：同时拥有结构校对和功能配准的神经元共有 {len(df_coreg_filtered)} 个！")

# 保存到缓存
    with open(COREG_CACHE, 'wb') as f:
        pickle.dump(df_coreg_filtered, f)
print(f"功能配准数据已缓存。")

# # 查看最终可以用于后续分析的数据长什么样
# # 我们重点关注 session, scan_idx, unit_id 和 pt_root_id
# display(df_coreg_filtered[['session', 'scan_idx', 'unit_id', 'pt_root_id']].head())

global_root_ids = df_coreg_filtered['pt_root_id'].drop_duplicates().tolist()
N_global = len(global_root_ids)
print(f"-> 准备构建的全局结构神经元总数：{N_global}")

with open(GLOBAL_IDS_CACHE, 'wb') as f:
    pickle.dump(global_root_ids, f)


import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 辅助函数：定义单个“工人”要干的活

def fetch_single_chunk(client, table_name, chunk, col_name, timestamp):
    """
    这是一个原子任务：仅仅负责向服务器索要这一个小 chunk 的数据并返回。
    """
    try:
        df_chunk = client.materialize.query_table(
            table_name,
            filter_in_dict={col_name: chunk},
            timestamp=timestamp
        )
        return df_chunk
    except Exception as e:
        print(f"\n[警告] 某个区块下载失败: {e}，将返回空表。")
        return pd.DataFrame() # 遇到错误返回空表，防止程序崩溃

# 主函数：并行调度中心
def fetch_synapses_parallel(client, table_name, root_ids, direction='post', chunk_size=50, max_workers=5):
    """
    利用线程池并行下载海量突触数据
    """
    col_name = 'post_pt_root_id' if direction == 'post' else 'pre_pt_root_id'
    total = len(root_ids)

    # 将长达 1451 个 ID 的列表，切割成多个长度为 50 的小碎片列表
    chunks = [root_ids[i:i + chunk_size] for i in range(0, total, chunk_size)]
    total_chunks = len(chunks)

    print(f"开始并行提取 ({direction}方向)...")
    print(f"总 ID 数: {total} | 切割成了 {total_chunks} 个任务块 | 启用了 {max_workers} 个并行线程")

    start_time = time.time()
    all_dfs = []

    # 启动线程池
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 将任务分发给线程池中的“工人”
        # executor.submit 的第一个参数是函数名，后面是传给该函数的参数
        futures = [executor.submit(fetch_single_chunk, client, table_name, chunk, col_name, timestamp) for chunk in chunks]

        # as_completed 会在某个“工人”完成任务时立刻捕捉到结果
        completed_count = 0
        for future in as_completed(futures):
            # 获取工人返回的 DataFrame
            df_result = future.result()
            if not df_result.empty:
                all_dfs.append(df_result)

            completed_count += 1
            # 打印进度条
            print(f"\r进度: [{completed_count}/{total_chunks}] 块已下载完成...", end="")

    print("\n正在拼接所有数据区块...")
    final_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    end_time = time.time()
    print(f"-> 并行提取大功告成！共下载了 {len(final_df)} 条突触记录。耗时: {end_time - start_time:.2f} 秒。\n")

    return final_df

# 执行并行下载
# 假设 global_root_ids 包含了之前的 1451 个 ID
# 将 max_workers 设置为 5 ，由于被对方服务器拦截，将 pre_synapses 的max worker改成 3
# 并行提取接收的输入突触并将其存入缓存
if os.path.exists(SYN_IN_CACHE):
    print("加载缓存的输入突触数据...")
    with open(SYN_IN_CACHE, 'rb') as f:
        synapses_in_df_fast = pickle.load(f)
else:
    print("开始并行下载输入突触...")
    synapses_in_df_fast = fetch_synapses_parallel(
        client, 'synapses_pni_2', global_root_ids, direction='post', chunk_size=40, max_workers=4
    )

# 保存输入突触缓存

with open(SYN_IN_CACHE, 'wb') as f:
    pickle.dump(synapses_in_df_fast, f)
print("输入突触数据已缓存。")

# 并行提取发送的输出突触
if os.path.exists(SYN_OUT_CACHE):
    print("加载缓存的输出突触数据...")
    with open(SYN_OUT_CACHE, 'rb') as f:
        synapses_out_df_fast = pickle.load(f)
else:
    print("开始并行下载输出突触...")
    synapses_out_df_fast = fetch_synapses_parallel(
        client, 'synapses_pni_2', global_root_ids, direction='pre', chunk_size=40, max_workers=3
    )
    with open(SYN_OUT_CACHE, 'wb') as f:
        pickle.dump(synapses_out_df_fast, f)

print("所有数据加载/下载完成！")