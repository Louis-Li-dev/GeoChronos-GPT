from tqdm import tqdm
import numpy as np
from copy import deepcopy


def split_data_by_value(data, keep_original=False):
    x = []
    y = []
    original_data = []
    for item in tqdm(data):
        unique_values = np.unique(item)[1:]  # Exclude the first unique value (typically 0)
        
        x_data = deepcopy(item)
        y_data = deepcopy(item)
        
        half_index = len(unique_values) // 2
        if half_index >= 2:    
            for idx, unique_val in enumerate(unique_values):
                val_positions = np.where(item == unique_val)
                
                if idx > half_index:
                    x_data[val_positions] = 0
                    y_data[val_positions] = 1
                else:
                    x_data[val_positions] = 1
                    y_data[val_positions] = 0
            if keep_original:
                original_data.append(item)
            x.append(x_data)
            y.append(y_data)
    return (x, y) if not keep_original else (x, y, original_data)