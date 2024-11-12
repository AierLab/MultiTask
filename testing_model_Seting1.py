import time,argparse,sys,os
import torch,math,random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
# from datasets.WSG_dataset import my_dataset_eval
from datasets.dataset_pairs_wRandomSample import my_dataset_eval
import torchvision.transforms as transforms
import matplotlib.image as img
from utils.UTILS import compute_psnr,compute_ssim


sys.path.append(os.getcwd())
# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()


parser.add_argument('--eval_in_path_Haze', type=str,default= '/mnt/pipeline_1/set1/rain_drop/test_a/data/')
parser.add_argument('--eval_gt_path_Haze', type=str,default= '/mnt/pipeline_1/set1/rain_drop/test_a/gt/')

parser.add_argument('--eval_in_path_Rain', type=str,default= '/mnt/pipeline_1/set1/rain/train/in/')
parser.add_argument('--eval_gt_path_Rain', type=str,default= '/mnt/pipeline_1/set1/rain/train/gt/')

parser.add_argument('--eval_in_path_L', type=str,default= '/mnt/pipeline_1/set1/snow/media/jdway/GameSSD/overlapping/test/Snow100K-L/synthetic/')
parser.add_argument('--eval_gt_path_L', type=str,default= '/mnt/pipeline_1/set1/snow/media/jdway/GameSSD/overlapping/test/Snow100K-L/gt/')

# parser.add_argument('--eval_in_path_realSnow', type=str,default= '/gdata2/zhuyr/Weather/Data/Snow/test_realistic/')
# parser.add_argument('--eval_in_path_realRain', type=str,default= '/gdata2/zhuyr/Weather/Data/RealRain300/')
# parser.add_argument('--eval_in_path_realRainDrop', type=str,default= '/gdata2/zhuyr/Weather/Data/RainDS/RainDS/RainDS_real/test_set/raindrop/')
# parser.add_argument('--eval_gt_path_realRainDrop', type=str,default= '/gdata2/zhuyr/Weather/Data/RainDS/RainDS/RainDS_real/test_set/gt/')
# /mnt/pipeline_1/MLT/Weather/training_try_stage2_share/net_epoch_119.pth
parser.add_argument('--model_path', type=str,default= '/home/4paradigm/Weather/stage1/')
parser.add_argument('--model_name', type=str,default= 'net_epoch_40.pth')
parser.add_argument('--save_path', type=str,default= '/mnt/pipeline_1/MLT/')

#training setting
parser.add_argument('--flag', type=str, default= 'O')
parser.add_argument('--base_channel', type = int, default= 18)
parser.add_argument('--num_block', type=int, default= 6)
args = parser.parse_args()
mask_A_dir = '/home/4paradigm/Weather/masks/maskA_epoch0.pth'
mask_B_dir = '/home/4paradigm/Weather/masks/maskB_epoch0.pth'
mask_C_dir = '/home/4paradigm/Weather/masks/maskC_epoch0.pth'
maskA = torch.load(mask_A_dir)
maskB = torch.load(mask_B_dir)
maskC = torch.load(mask_C_dir)
model_ori ='/home/4paradigm/Weather/stage1/net_epoch_99.pth'
model_mask='/mnt/pipeline_2/MLT/training_tune_percent90_mask/net_epoch_8.pth'


def load_combined_model(net, model_path1, model_path2, mask):
    # 加载两个预训练模型的 state_dict
    state_dict1 = torch.load(model_path1)
    state_dict2 = torch.load(model_path2)
    
    # 初始化组合后的 state_dict
    combined_state_dict = {}
    
    # 遍历 state_dict1，将参数根据 mask 进行组合
    with torch.no_grad():
        for name, param1 in state_dict1.items():
            # 如果 state_dict2 中没有该参数，则使用 state_dict1 的参数
            if name not in state_dict2:
                combined_state_dict[name] = param1
            else:
                param2 = state_dict2[name]
                # 根据 mask 的值选择来自 param1 或 param2 的参数
                if name in mask:
                    combined_state_dict[name] = torch.where(mask[name].bool(), param1, param2)
                else:
                    combined_state_dict[name] = param1  # 如果 mask 中没有该参数，则直接选择 param1
    
    # 将组合后的参数加载到模型中
    net.load_state_dict(combined_state_dict, strict=False)
    return net




trans_eval = transforms.Compose(
        [
         transforms.ToTensor()
        ])

if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
    
def get_eval_data(val_in_path=args.eval_in_path_L,val_gt_path =args.eval_gt_path_L ,trans_eval=trans_eval):
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label =val_gt_path, transform=trans_eval,fix_sample= 500 )
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4)
    return eval_loader

def test(net,eval_loader,Dname = 'S',flag = [1,0,0], model_flag= args.flag, save_results_path=args.save_path):
    net.to('cuda:0')
    net.eval()
    st = time.time()
    with torch.no_grad():
        eval_output_psnr = 0.0
        eval_input_psnr = 0.0
        eval_output_ssim = 0.0
        eval_input_ssim = 0.0
        
        final_save_path = save_results_path + '-setting1-'+ Dname+'/'
        if not os.path.exists(final_save_path):
            os.mkdir(final_save_path)
        for index, (data_in, label, name) in enumerate(tqdm(eval_loader), 0):# enumerate(eval_loader, 0):
            inputs = Variable(data_in).to('cuda:0')
            labels = Variable(label).to('cuda:0')
            # import pdb;pdb.set_trace()

            if model_flag == 'S1':
                outputs = net(inputs)
            else:
                outputs = net(inputs, flag=flag)
            eval_input_psnr += compute_psnr(inputs, labels)
            eval_output_psnr += compute_psnr(outputs, labels)

            eval_input_ssim += compute_ssim(inputs, labels)
            eval_output_ssim += compute_ssim(outputs, labels)
            
            # out_eval_np = np.squeeze(torch.clamp(outputs, 0., 1.).cpu().detach().numpy()).transpose((1,2,0))
            # # img.imsave(final_save_path+ name[0], np.uint8(out_eval_np * 255.))
            

        Final_output_PSNR = eval_output_psnr / len(eval_loader)
        Final_input_PSNR = eval_input_psnr / len(eval_loader)
        Final_output_SSIM = eval_output_ssim / len(eval_loader)
        Final_input_SSIM = eval_input_ssim / len(eval_loader)

        print("Dname:{}--------------[Num_eval:{} In_PSNR:{}  "
              "Out_PSNR:{},In_SSIM:{}  Out_SSIM:{}]:-----cost time;{}".format(Dname,len(eval_loader),
                round(Final_input_PSNR,5),round(Final_output_PSNR, 5),
                round(Final_input_SSIM, 5),round(Final_output_SSIM, 5),time.time() -st))
def print_indictor(indictor):
    indictor_list = []
    for i in range(len(indictor)):
        indictor_list.append(indictor[i].item())
    indictor_array = np.array(indictor_list)
    print('indictor_array---ori:',list(indictor_array)) #indictor_array)

    x = np.zeros_like(indictor_array)
    y = np.ones_like(indictor_array)
    out = np.where(indictor_array>0.1, y,x)
    print('indictor_array---Binary out:',list(out))
    
if __name__ == '__main__':
    if args.flag == 'O':
        # from networks.Network_Stage2_K1_Flag import UNet
        from networks.Network_our import UNet
        print("network is ours")
    elif args.flag == 'K3':
        from networks.Network_Stage2_K3_Flag import UNet
    elif args.flag == 'S1':
        from networks.Network_Stage1 import UNet

    net = UNet(base_channel=args.base_channel, num_res=args.num_block)
    

    index = 0

    # model_name = args.model_name
    # pretrained_model = torch.load(args.model_path + model_name)
    # net.load_state_dict(pretrained_model, strict=False)
    net_A = load_combined_model(net,model_mask,model_ori,maskA)
    net_B = load_combined_model(net,model_mask,model_ori,maskA)
    net_C = load_combined_model(net,model_mask,model_ori,maskA)
    print('----Load successfully!------')

    if args.flag != 'S1':
        indictor1 = net.getIndicators_B1()
        indictor2 = net.getIndicators_B2()
        indictor3 = net.getIndicators_B3()


    
        
    # Datasets of Setting1
    eval_loader_Haze = get_eval_data(val_in_path=args.eval_in_path_Haze, val_gt_path=args.eval_gt_path_Haze)
    eval_loader_L = get_eval_data(val_in_path=args.eval_in_path_L, val_gt_path=args.eval_gt_path_L)
    eval_loader_Rain = get_eval_data(val_in_path=args.eval_in_path_Rain, val_gt_path=args.eval_gt_path_Rain)
    #Datasets of Real-world Scenes
    # eval_loader_RealRain = get_eval_data(val_in_path=args.eval_in_path_realRain, val_gt_path=args.eval_in_path_realRain)
    # eval_loader_RealSnow = get_eval_data(val_in_path=args.eval_in_path_realSnow, val_gt_path=args.eval_in_path_realSnow)
    # eval_loader_RealRainDrop = get_eval_data(val_in_path=args.eval_in_path_realRainDrop, val_gt_path=args.eval_gt_path_realRainDrop)


    # # RainDrop
    test(net=net_C, eval_loader = eval_loader_Haze, Dname= 'RD',flag = [0,0,1],model_flag= args.flag)
    # # OutDoor-Rain
    test(net=net_B, eval_loader = eval_loader_Rain, Dname= 'HRain',flag = [0,1,0],model_flag= args.flag)
    # # Snow
    test(net=net_A, eval_loader = eval_loader_L,  Dname= 'L',flag = [1,0,0],model_flag= args.flag)
    
    # test(net=net, eval_loader = eval_loader_RealSnow, Dname= 'RealSnow',flag = [1,0,0],model_flag= args.flag)
    # test(net=net, eval_loader = eval_loader_RealRain,Dname= 'RealRain',flag = [0,1,0],model_flag= args.flag)
    # test(net=net, eval_loader = eval_loader_RealRainDrop,Dname= 'RealRainDrop',flag = [0,0,1],model_flag= args.flag)


