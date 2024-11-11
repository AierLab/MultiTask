import time
import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch import random
from torch.autograd import Variable
import time,argparse,sys,os
import torch,math,random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
# from utils.UTILS import compute_psnr, compute_ssim  # 确保你的utils模块包含这些函数
from datasets.dataset_pairs_wRandomSample import my_dataset_eval
import matplotlib.image as img

trans_eval = transforms.Compose(
        [
         transforms.ToTensor()
        ])

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_eval_data(val_in_path, val_gt_path ,trans_eval=trans_eval):
    eval_data = my_dataset_eval(root_in=val_in_path, root_label=val_gt_path, transform=trans_eval, fix_sample=500)
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4)
    return eval_loader

def inference(net, eval_loader, save_results_path):
    net.eval()
    st = time.time()
    with torch.no_grad():
        for index, (data_in, _, name) in enumerate(eval_loader):
            inputs = Variable(data_in).to('cuda:0')
            outputs = net(inputs)
            
            # 处理输出结果（例如，保存图像）
            out_eval_np = np.squeeze(torch.clamp(outputs, 0., 1.).cpu().detach().numpy()).transpose((1, 2, 0))
            img.imsave(os.path.join(save_results_path, f"output_{index}.png"), np.uint8(out_eval_np * 255.))
            
            print(f"Processed image: {name[0]} in {time.time() - st:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_path', type=str, required=True, help='Path to input images for inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--save_path', type=str, default='/home/4paradigm/Weather/img_save', help='Path to save output images')
    parser.add_argument('--base_channel', type=int, default=18, help='Base channel for UNet')
    parser.add_argument('--num_block', type=int, default=6, help='Number of blocks in UNet')
    parser.add_argument('--flag', type=str, default='S1', help='Model flag (O, K1, K3)')
    
    
    parser.add_argument('--eval_in_path_Haze', type=str,default= '/mnt/pipeline_1/set1/rain_drop/test_a/data/')
    parser.add_argument('--eval_gt_path_Haze', type=str,default= '/mnt/pipeline_1/set1/rain_drop/test_a/gt/')

    parser.add_argument('--eval_in_path_Rain', type=str,default= '/mnt/pipeline_1/set1/rain/train/in/')
    parser.add_argument('--eval_gt_path_Rain', type=str,default= '/mnt/pipeline_1/set1/rain/train/gt/')

    parser.add_argument('--eval_in_path_L', type=str,default= '/mnt/pipeline_1/set1/snow/media/jdway/GameSSD/overlapping/test/Snow100K-L/synthetic/')
    parser.add_argument('--eval_gt_path_L', type=str,default= '/mnt/pipeline_1/set1/snow/media/jdway/GameSSD/overlapping/test/Snow100K-L/gt/')
    args = parser.parse_args()

    # 设置随机种子
    setup_seed(20)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    if args.flag == 'S1':
        from networks.Network_Stage1 import UNet
    elif args.flag == 'O':
        from networks.Network_our import UNet


    net = UNet(base_channel=args.base_channel, num_res=args.num_block)
    pretrained_model = torch.load(args.model_path)
    import pdb;pdb.set_trace()
    net.load_state_dict(pretrained_model, strict=True)
    net.to(device)
    print('Model loaded successfully!')

    # 图像预处理
    trans_eval = transforms.Compose([
        transforms.ToTensor()
    ])

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # 获取评估数据
    eval_loader_Haze = get_eval_data(val_in_path=args.eval_in_path_Haze, val_gt_path=args.eval_gt_path_Haze)
    eval_loader_L = get_eval_data(val_in_path=args.eval_in_path_L, val_gt_path=args.eval_gt_path_L)
    eval_loader_Rain = get_eval_data(val_in_path=args.eval_in_path_Rain, val_gt_path=args.eval_gt_path_Rain)
    # eval_loader = get_eval_data(args.input_path, trans_eval)

    # 推断
    inference(net, eval_loader_Haze, args.save_path)
