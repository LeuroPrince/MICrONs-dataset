import os
import pandas as pd
import numpy as np

CACHE_DIR = './cache'
with open(os.path.join(CACHE_DIR, 'W_sorted.npy'), 'rb') as f:
    W_sorted = np.load(f)

# Create labels for rows and columns based on the sorted order and broad layer
# sorting_df already contains the global_root_ids and their broad_layers in the correct order
neuron_labels = [f"{row.broad_layer}_{row.pt_root_id}" for index, row in sorting_df.iterrows()]

# Convert the W_sorted numpy array to a pandas DataFrame
W_sorted_df = pd.DataFrame(W_sorted, index=neuron_labels, columns=neuron_labels)

# Save the DataFrame to a CSV file
output_filename = 'W_sorted_with_layers.csv'
W_sorted_df.to_csv(output_filename)

print(f"'W_sorted' 矩阵已成功保存到 '{output_filename}'，并带有层级和神经元ID标签。")

# Display the first few rows and columns of the DataFrame to verify
W_sorted_df.head()