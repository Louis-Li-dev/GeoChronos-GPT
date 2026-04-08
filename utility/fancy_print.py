import matplotlib.pyplot as plt

def print_fancy_console(target_region, width, height, model_names, pooling_mode=None, 
                       pool_size=None, stride=None):
    """Print a modern, compact, dynamic console banner for model evaluation with pooling info."""
    # Compute dynamic width
    line_items = [f"Target Region: {target_region}",
                  f"Output Dimensions: {width} × {height}",
                  f"Models Evaluated: {', '.join(model_names)}"]
    if pooling_mode:
        pooling_info = f"Pooling: {pooling_mode}"
        if pool_size:
            pooling_info += f", Size: {pool_size}"
        if stride:
            pooling_info += f", Stride: {stride}"
        line_items.insert(0, pooling_info)

    box_width = max(len(line) for line in line_items) + 6
    border_top = f"╔{'═' * (box_width - 2)}╗"
    border_mid = f"╟{'─' * (box_width - 2)}╢"
    border_bot = f"╚{'═' * (box_width - 2)}╝"

    print()
    print(border_top)
    print(f"║ {'Model Evaluation'.center(box_width - 4)} ║")
    print(border_mid)
    for line in line_items:
        print(f"║ {line.ljust(box_width - 4)} ║")
    print(border_bot)
    print()
    

def draw_Hit_Frequency_Histogram(dt,interval,Max,name):
    
    
    Max = ((Max/interval)+(Max%interval!=0))*interval
        
    data = [0 for i in range(Max/interval)]
    
    for u,v in dt.items():
        data[v/interval+(v%interval!=0)]+=1
    

    # 建立對應的 x 軸標籤，例如 '1-10', '11-20', ..., '191-200'
    labels = [f'{i}-{i+interval-1}' for i in range(1, Max, interval)]  # step為10

    # 畫圖
    plt.figure(figsize=(12, 6))
    plt.bar(labels, data, color='skyblue')
    plt.xlabel('旅遊景點出現次數區間')
    plt.ylabel('旅遊景點數量')
    plt.title('旅遊景點出現次數數量分佈')
    plt.xticks(rotation=45)  # 旋轉 x 軸文字以避免重疊
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{name}.jpg", dpi=300, quality=95)
