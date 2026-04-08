import math
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 4})


def slice_for_draw(dt,interval,Max):
    Max = ((Max/interval)+(Max%interval!=0))*interval
    
    data = [0 for i in range(math.ceil(Max/interval))]
    
    for u,v in dt.items():
        # print(f'{u},{v}')
        data[int(v/interval+(v%interval!=0))-1]+=1
        
    return data

def draw_HFH(dt,interval,Max,name,K):
    fig, axes = plt.subplots(3, 2)  # 建立 2x2 的子圖
    
    all_key = list(dt.keys())
    
    labels = [f'{i}-{i+interval-1}' for i in range(1, Max, interval)]  # step為10
    
    for i in range(6):
        who = all_key[i]
        print(dt[who]['data'])
        print(labels)
        axes[i//2,i%2].bar(labels, dt[who]['data'], color='skyblue')
        axes[i//2,i%2].set_xlabel('frequecy for Tourist attraction')
        axes[i//2,i%2].set_ylabel('counts for frequency')
        axes[i//2,i%2].tick_params(axis='x', labelrotation=45)
        axes[i//2,i%2].set_title(dt[who]['name'])
    
    plt.suptitle(f"nDCG@{K},Hit_Frequency_Histogram", fontsize=8)
    plt.tight_layout()  # 自動調整排版
    plt.savefig(f"photos\{name}.jpg", dpi=300)
    plt.show()
    plt.close()