import os
import re
import matplotlib.pyplot as plt

# 定义一个函数来从日志文件中提取数据
def parse_log(file_path):
    data = {
        'epoch': [],
        'avg_lossA': [],
        'avg_lossB': [],
        'avg_lossC': [],
        'avg_loss': [],
        'in_PSNR_A': [],
        'out_PSNR_A': [],
        'in_PSNR_B': [],
        'out_PSNR_B': [],
        'in_PSNR_C': [],
        'out_PSNR_C': []
    }
    
    # 正则表达式提取所需的信息
    pattern = re.compile(
        r'\[epoch:(\d+) / EPOCH :\d+\].*?'
        r'avg_lossA:([\d.]+), avg_lossB:([\d.]+), avg_lossC:([\d.]+), avg_loss:([\d.]+).*?'
        r'in_PSNR_A: ([\d.]+), out_PSNR_A: ([\d.]+).*?'
        r'in_PSNR_B: ([\d.]+), out_PSNR_B: ([\d.]+).*?'
        r'in_PSNR_C: ([\d.]+), out_PSNR_C: ([\d.]+)'
    )
    
    # 读取日志文件
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                data['epoch'].append(int(match.group(1)))
                data['avg_lossA'].append(float(match.group(2)))
                data['avg_lossB'].append(float(match.group(3)))
                data['avg_lossC'].append(float(match.group(4)))
                data['avg_loss'].append(float(match.group(5)))
                data['in_PSNR_A'].append(float(match.group(6)))
                data['out_PSNR_A'].append(float(match.group(7)))
                data['in_PSNR_B'].append(float(match.group(8)))
                data['out_PSNR_B'].append(float(match.group(9)))
                data['in_PSNR_C'].append(float(match.group(10)))
                data['out_PSNR_C'].append(float(match.group(11)))
                
    return data

# 绘制损失和PSNR曲线并保存图像
def plot_metrics(data, save_path='metrics_plot.png'):
    epochs = data['epoch']
    
    # 设置子图
    fig, axs = plt.subplots(2, 1, figsize=(12, 18))
    fig.suptitle("Training Metrics")
    
    # 绘制 Loss 图
    axs[0].plot(epochs, data['avg_lossA'], label='avg_lossA', marker='o')
    axs[0].plot(epochs, data['avg_lossB'], label='avg_lossB', marker='o')
    axs[0].plot(epochs, data['avg_lossC'], label='avg_lossC', marker='o')
    axs[0].plot(epochs, data['avg_loss'], label='avg_loss', marker='o')
    axs[0].set_title("Average Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)
    
    # 绘制 PSNR_A 图
    # axs[1].plot(epochs, data['in_PSNR_A'], label='in_PSNR_A', marker='x')
    axs[1].plot(epochs, data['out_PSNR_A'], label='out_PSNR_A', marker='x')
    axs[1].plot(epochs, data['out_PSNR_B'], label='out_PSNR_B', marker='x')
    # axs[2].plot(epochs, data['in_PSNR_C'], label='in_PSNR_C', marker='x')
    axs[1].plot(epochs, data['out_PSNR_C'], label='out_PSNR_C', marker='x')
    axs[1].set_title("PSNR for A, B and C")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("PSNR")
    axs[1].legend()
    axs[1].grid(True)
    
    # # 绘制 PSNR_B 和 PSNR_C 图
    # # axs[2].plot(epochs, data['in_PSNR_B'], label='in_PSNR_B', marker='x')
    # axs[2].plot(epochs, data['out_PSNR_B'], label='out_PSNR_B', marker='x')
    # # axs[2].plot(epochs, data['in_PSNR_C'], label='in_PSNR_C', marker='x')
    # axs[2].plot(epochs, data['out_PSNR_C'], label='out_PSNR_C', marker='x')
    # axs[2].set_title("PSNR for B and C")
    # axs[2].set_xlabel("Epoch")
    # axs[2].set_ylabel("PSNR")
    # axs[2].legend()
    # axs[2].grid(True)
    
    # 调整布局并保存图像
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")

# 主程序
if __name__ == "__main__":
    log_file = '/home/4paradigm/WGWS-Net/log/output_80_completed.txt'  # 日志文件路径
    log_file_name = os.path.splitext(os.path.basename(log_file))[0]

    data = parse_log(log_file)
    plot_metrics(data, save_path=f'/home/4paradigm/WGWS-Net/metrics/{log_file_name}_metrics.png')
