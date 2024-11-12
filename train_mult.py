from datetime import datetime
import os,cv2,time,torchvision,argparse,logging,sys,os,gc
import shutil
import torch,math,random
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.autograd import Variable
import torch.optim as optim
from datasets.WSG_dataset import my_dataset
from datasets.dataset_pairs_wRandomSample import my_dataset_eval
# from datasets.dataset_pairs_wRandomSample import my_dataset,my_dataset_eval
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,MultiStepLR
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.UTILS import compute_psnr
import loss.losses as losses
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from loss.perceptual import LossNetwork
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.getcwd())
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

if torch.cuda.device_count() ==6:
    MULTI_GPU = True
    print('MULTI_GPU {}:'.format(torch.cuda.device_count()), MULTI_GPU )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
    device_ids = [0, 1,2,3,4,5]
if torch.cuda.device_count() == 4:
    MULTI_GPU = True
    print('MULTI_GPU {}:'.format(torch.cuda.device_count()), MULTI_GPU )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1,2,3"
    device_ids = [0, 1,2,3]
if torch.cuda.device_count() == 2:
    MULTI_GPU = True
    print('MULTI_GPU {}:'.format(torch.cuda.device_count()), MULTI_GPU )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    device_ids = [0, 1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device ----------------------------------------:',device)

parser = argparse.ArgumentParser()
# path setting
parser.add_argument('--experiment_name', type=str,default= "training_tune_percent95_mask_ori") # modify the experiments name-->modify all save path
parser.add_argument('--unified_path', type=str,default=  '/mnt/pipeline_2/MLT/')
#parser.add_argument('--model_save_dir', type=str, default= )#required=True
parser.add_argument('--training_in_path', type=str,default= '/mnt/pipeline_1/set1/snow/all/synthetic/')
parser.add_argument('--training_gt_path', type=str,default= '/mnt/pipeline_1/set1/snow/all/gt/')

parser.add_argument('--training_in_pathRain', type=str,default= '/mnt/pipeline_1/set1/rain/train/in/')
parser.add_argument('--training_gt_pathRain', type=str,default= '/mnt/pipeline_1/set1/rain/train/gt/')

parser.add_argument('--training_in_pathRD', type=str,default= '/mnt/pipeline_1/set1/rain_drop/train/data/')#  RainDrop 1110 pairs
parser.add_argument('--training_gt_pathRD', type=str,default= '/mnt/pipeline_1/set1/rain_drop/train/gt/')


parser.add_argument('--writer_dir', type=str, default= '/mnt/pipeline_1/MLT/writer_logs/')
parser.add_argument('--logging_path', type=str, default= '/mnt/pipeline_1/MLT/logging/')

parser.add_argument('--eval_in_path_RD', type=str,default= '/mnt/pipeline_1/set1/rain_drop/test_a/data/')
parser.add_argument('--eval_gt_path_RD', type=str,default= '/mnt/pipeline_1/set1/rain_drop/test_a/gt/')

parser.add_argument('--eval_in_path_L', type=str,default= '/mnt/pipeline_1/set1/snow/media/jdway/GameSSD/overlapping/test/Snow100K-L/synthetic/')
parser.add_argument('--eval_gt_path_L', type=str,default= '/mnt/pipeline_1/set1/snow/media/jdway/GameSSD/overlapping/test/Snow100K-L/gt/')

parser.add_argument('--eval_in_path_Rain', type=str,default= '/mnt/pipeline_1/set1/rain/train/in/')
parser.add_argument('--eval_gt_path_Rain', type=str,default= '/mnt/pipeline_1/set1/rain/train/gt/')
#training setting
parser.add_argument('--EPOCH', type=int, default= 180)
parser.add_argument('--T_period', type=int, default= 60)  # CosineAnnealingWarmRestarts
parser.add_argument('--BATCH_SIZE', type=int, default= 10)
parser.add_argument('--Crop_patches', type=int, default= 256)
parser.add_argument('--learning_rate', type=float, default= 0.0001)
parser.add_argument('--print_frequency', type=int, default= 200)
parser.add_argument('--SAVE_Inter_Results', type=bool, default= False)
#during training
parser.add_argument('--max_psnr', type=int, default= 25)
parser.add_argument('--fix_sample', type=int, default= 10000)
parser.add_argument('--VGG_lamda', type=float, default= 0.1)

parser.add_argument('--debug', type=bool, default= False)
parser.add_argument('--lam', type=float, default= 0.1)
parser.add_argument('--flag', type=str, default= 'K1')
parser.add_argument('--pre_model', type=str,default= '/home/4paradigm/Weather/stage1/net_epoch_99.pth')

#training setting
parser.add_argument('--base_channel', type = int, default= 20)
parser.add_argument('--num_block', type=int, default= 6)
parser.add_argument('--world_size', default=4, type=int, help='number of distributed processes')
parser.add_argument('--rank', type=int, help='rank of distributed processes')
args = parser.parse_args()


if args.debug == True:
    fix_sample = 200
else:
    fix_sample = args.fix_sample

exper_name =args.experiment_name
writer = SummaryWriter(args.writer_dir + exper_name)
if not os.path.exists(args.writer_dir):
    os.makedirs(args.writer_dir, exist_ok=True)
if not os.path.exists(args.logging_path):
    os.makedirs(args.logging_path, exist_ok=True)

unified_path = args.unified_path
SAVE_PATH =unified_path  + exper_name + '/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH, exist_ok=True)
if args.SAVE_Inter_Results:
    SAVE_Inter_Results_PATH = unified_path + exper_name +'__inter_results/'
    if not os.path.exists(SAVE_Inter_Results_PATH):
        os.makedirs(SAVE_Inter_Results_PATH, exist_ok=True)

base_channel=args.base_channel
num_res = args.num_block

trans_eval = transforms.Compose(
        [
         transforms.ToTensor()
        ])
logging.info(f'begin testing! ')
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def save_masked_model(net, mask, original_parameters, epoch, suffix):
    # 确保路径 SAVE_PATH + suffix 存在
    save_path = os.path.join(SAVE_PATH, suffix)
    os.makedirs(save_path, exist_ok=True)
    
    # 使用掩码更新参数
    with torch.no_grad():
        for param, original_param, m in zip(net.parameters(), original_parameters, mask):
            # param.data = param.data*m + original_param.data*(1-m)
            param.data = m * param.data + (1 - m.float()) * original_param.data
            # param.data = param.data * mask + original_param.data * (1 - mask.float())
            # param.data = torch.where(m.bool(), param.data, original_param.data)

    # 保存更新后的模型
    save_model = os.path.join(save_path, f'net_epoch_{epoch}.pth')
    torch.save(net.state_dict(), save_model)
    return save_model




def test(net,eval_loader, save_model ,epoch =1,max_psnr_val=26 ,Dname = 'S',flag = [1,0,0]):
    net.to('cuda:0')
    net.eval()
    torch.distributed.barrier() 
    net.load_state_dict(torch.load(save_model), strict=True)

    st = time.time()
    with torch.no_grad():
        eval_output_psnr = 0.0
        eval_input_psnr = 0.0
        for index, (data_in, label, name) in enumerate(eval_loader, 0):#enumerate(tqdm(eval_loader), 0):
            inputs = Variable(data_in).to('cuda:0')
            labels = Variable(label).to('cuda:0')

            outputs = net(inputs)
            eval_input_psnr += compute_psnr(inputs, labels)
            eval_output_psnr += compute_psnr(outputs, labels)
        Final_output_PSNR = eval_output_psnr / len(eval_loader)
        Final_input_PSNR = eval_input_psnr / len(eval_loader)
        writer.add_scalars(exper_name + '/testing', {'eval_PSNR_Output': eval_output_psnr / len(eval_loader),
                                                     'eval_PSNR_Input': eval_input_psnr / len(eval_loader), }, epoch)
        if Final_output_PSNR > max_psnr_val:  #just save better model
            max_psnr_val = Final_output_PSNR
        print("epoch:{}---------Dname:{}--------------[Num_eval:{} In_PSNR:{}  Out_PSNR:{}]--------max_psnr_val:{}:-----cost time;{}".format(epoch, Dname,len(eval_loader),round(Final_input_PSNR, 2),
                                                                                        round(Final_output_PSNR, 2), round(max_psnr_val, 2),time.time() -st))
    return max_psnr_val


def save_imgs_for_visual(path,inputs,labels,outputs):
    torchvision.utils.save_image([inputs.cpu()[0], labels.cpu()[0], outputs.cpu()[0]], path,nrow=3, padding=0)

def get_training_data(fix_sample=fix_sample, Crop_patches=args.Crop_patches):
    # A:snow100     B:outdoor_rain      C:raindrop
    rootA_in = args.training_in_path
    rootA_label = args.training_gt_path
    rootA_txt = '/mnt/pipeline_1/set1/data_txt/train/snow_images.txt'
    rootB_in = args.training_in_pathRain
    rootB_label = args.training_gt_pathRain
    rootB_txt = '/mnt/pipeline_1/set1/data_txt/train/rain.txt'
    rootC_in = args.training_in_pathRD
    rootC_label = args.training_gt_pathRD
    rootC_txt = '/mnt/pipeline_1/set1/data_txt/train/raindrop_images.txt'
    train_datasets = my_dataset(rootA_in, rootA_label,rootA_txt,rootB_in, rootB_label,rootB_txt,rootC_in, rootC_label,rootC_txt,crop_size =Crop_patches,
                                fix_sample_A = fix_sample, fix_sample_B = fix_sample,fix_sample_C = fix_sample)
    # train_loader = DataLoader(dataset=train_datasets, batch_size=args.BATCH_SIZE, num_workers= 6 ,shuffle=True)
    # print('len(train_loader):' ,len(train_loader))
    # return train_loader
    return train_datasets

def get_eval_data(val_in_path=args.eval_in_path_L,val_gt_path =args.eval_gt_path_L ,trans_eval=trans_eval):
    eval_data = my_dataset_eval(
        root_in=val_in_path, root_label =val_gt_path, transform=trans_eval,fix_sample= 500 )
    eval_loader = DataLoader(dataset=eval_data, batch_size=1, num_workers=4)
    return eval_loader
def print_param_number(net):
    print('#generator parameters:', sum(param.numel() for param in net.parameters()))
    Total_params = 0
    Trainable_params = 0

    for param in net.parameters():
        mulValue = np.prod(param.size())
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')

# Calculate threshold for smallest 20% in each gradient
# def calculate_mask(grad, percentage=20):
#     flattened_grad = torch.cat([g.view(-1) for g in grad])
#     threshold = torch.quantile(flattened_grad.abs(), percentage / 100.0)
#     mask = [torch.abs(g) < threshold for g in grad]
#     return mask

def ensure_consistent_mask(mask_list):
    """确保 mask 列表中的每个 tensor 元素与前 3 个 tensor 的相应位置保持一致（如果前三个相同）。"""
    for i in range(3, len(mask_list)):
        # 获取前 3 个 mask 的相应位置的 tensor（列表中的元素为list，每个list元素为tensor）
        prev1, prev2, prev3 = mask_list[i - 3], mask_list[i - 2], mask_list[i - 1]
        current = mask_list[i]
        # print(prev1[0].shape)  # Just a sample print for debugging, remove in production

        # 找到前 3 个 tensor 中相同的位置，逐元素比较
        consistent_mask_list = []
        for t1, t2, t3 in zip(prev1, prev2, prev3):
            # 对每对 tensor 使用 torch.eq() 进行逐元素比较
            consistent_mask = (t1 == t2) & (t2 == t3)
            consistent_mask_list.append(consistent_mask)

        # 更新 mask_list 的当前元素
        updated_mask_list = []
        for idx, (consistent_mask, t1,c1) in enumerate(zip(consistent_mask_list, prev1,current)):
            # 在 consistent_mask 对应位置为 True 时，比较 current[idx] 和 prev1[idx]
            updated_mask = torch.where(
                consistent_mask, 
                torch.where(c1 == t1, current[idx], ~current[idx]),  # 如果一致则保留 current，否则取反
                current[idx]  # 如果不一致，保留 consistent_mask 的值
            )
            updated_mask_list.append(updated_mask)

        # 更新 mask_list 的当前元素
        mask_list[i] = updated_mask_list

    return mask_list


def process_masks(maskAs, maskBs, maskCs):
    """处理 mask 列表并确保它们满足一致性条件。"""
    if len(maskAs) < 4:
        # 如果长度小于 4，将 maskAs 的所有元素复制到 maskB_change 和 maskC_change
        maskA_change = maskAs
        maskB_change = maskAs
        maskC_change = maskAs
    else:
        # 对 maskAs、maskBs 和 maskCs 应用一致性检查
        maskA_change = ensure_consistent_mask(maskAs)
        maskB_change = ensure_consistent_mask(maskBs)
        maskC_change = ensure_consistent_mask(maskCs)

    return maskA_change[-1], maskB_change[-1], maskC_change[-1]




def calculate_mask(grad, percentage=20):
    if grad is None or len(grad) == 0:
        return None
    
    # Initialize an empty list to hold masks
    masks = []

    for g in grad:
        if g is None:
            masks.append(False)
            continue
        
        flattened_grad = g.view(-1)
        threshold = torch.quantile(flattened_grad.abs(), percentage / 100.0)
        mask = torch.abs(g) > threshold
        
        masks.append(mask)

    return masks

# Ensure no overlap among maskA, maskB, and maskC
# def remove_overlap_masks(maskA, maskB, maskC):
#     # Create overlap areas
#     overlap_AB = [mA & mB for mA, mB in zip(maskA, maskB)]
#     overlap_AC = [mA & mC for mA, mC in zip(maskA, maskC)]
#     overlap_BC = [mB & mC for mB, mC in zip(maskB, maskC)]
#     overlap_ABC = [mA & mB & mC for mA, mB, mC in zip(maskA, maskB, maskC)]
#     print(len(maskA))
#     print(f"maskA is {maskA[0].shape}")
#     print(len(overlap_AB))
#     print(f"the overleap is {overlap_AB[0].shape}")
    
#     # Remove overlaps by setting to False where there's any overlap
#     for i in range(len(maskA)):
#         maskA[i] = torch.tensor(maskA[i]) & ~overlap_AB[i] & ~overlap_AC[i] & ~overlap_ABC[i]
#         maskB[i] = torch.tensor(maskB[i]) & ~overlap_AB[i] & ~overlap_BC[i] & ~overlap_ABC[i]
#         maskC[i] = torch.tensor(maskC[i]) & ~overlap_AC[i] & ~overlap_BC[i] & ~overlap_ABC[i]

def remove_overlap_masks(maskA, maskB, maskC):
    # Ensure masks are of the same length
    if len(maskA) != len(maskB) or len(maskA) != len(maskC):
        raise ValueError("All masks must have the same length.")

    # Create overlap areas
    overlap_AB = []
    overlap_AC = []
    overlap_BC = []
    overlap_ABC = []

    for mA, mB, mC in zip(maskA, maskB, maskC):
        if mA is None:
            mA = torch.ones_like(mB, dtype=torch.bool) if mB is not None else torch.empty(0, dtype=torch.bool)
        if mB is None:
            mB = torch.ones_like(mA, dtype=torch.bool) if mA is not None else torch.empty(0, dtype=torch.bool)
        if mC is None:
            mC = torch.ones_like(mA, dtype=torch.bool) if mA is not None else torch.empty(0, dtype=torch.bool)

        overlap_AB.append(mA & mB)
        overlap_AC.append(mA & mC)
        overlap_BC.append(mB & mC)
        overlap_ABC.append(mA & mB & mC)

    # Remove overlaps by setting to False where there's any overlap
    for i in range(len(maskA)):
        if maskA[i] is None:
            continue  # Skip if maskA[i] is None

        overlap_maskA = overlap_AB[i] if overlap_AB[i] is not None else torch.zeros_like(maskA[i], dtype=torch.bool)
        overlap_maskAC = overlap_AC[i] if overlap_AC[i] is not None else torch.zeros_like(maskA[i], dtype=torch.bool)
        overlap_maskABC = overlap_ABC[i] if overlap_ABC[i] is not None else torch.zeros_like(maskA[i], dtype=torch.bool)

        maskA[i] = maskA[i] & ~overlap_maskA & ~overlap_maskAC & ~overlap_maskABC

        if maskB[i] is not None:
            overlap_maskB = overlap_AB[i] if overlap_AB[i] is not None else torch.zeros_like(maskB[i], dtype=torch.bool)
            overlap_maskBC = overlap_BC[i] if overlap_BC[i] is not None else torch.zeros_like(maskB[i], dtype=torch.bool)
            maskB[i] = maskB[i] & ~overlap_maskB & ~overlap_maskBC & ~overlap_maskABC

        if maskC[i] is not None:
            overlap_maskC = overlap_AC[i] if overlap_AC[i] is not None else torch.zeros_like(maskC[i], dtype=torch.bool)
            overlap_maskBC = overlap_BC[i] if overlap_BC[i] is not None else torch.zeros_like(maskC[i], dtype=torch.bool)
            maskC[i] = maskC[i] & ~overlap_maskC & ~overlap_maskBC & ~overlap_maskABC


# Generate a time flag based on the current date and time
time_flag = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create the folder path using the time flag
folder = f"/mnt/pipeline_1/mask_log/run{time_flag}"
    
# Define a function to save masks to the local directory
# def save_masks_to_local(maskA, maskB, maskC, epoch, folder=folder):
#     # Ensure the folder exists
#     os.makedirs(folder, exist_ok=True)
    
#     # Save the masks
#     np.save(os.path.join(folder, f"maskAs_epoch{epoch}.npy"), maskA.cpu().numpy())
#     np.save(os.path.join(folder, f"maskBs_epoch{epoch}.npy"), maskB.cpu().numpy())
#     np.save(os.path.join(folder, f"maskCs_epoch{epoch}.npy"), maskC.cpu().numpy())
    
def save_masks_to_local(maskA, maskB, maskC, epoch, folder='/home/4paradigm/Weather/masks'):

    # 保存 maskAs
    os.makedirs(folder, exist_ok=True)
    torch.save(maskA, os.path.join(folder, f'maskA_epoch{epoch}.pth'))
    torch.save(maskB, os.path.join(folder, f'maskB_epoch{epoch}.pth'))
    torch.save(maskC, os.path.join(folder, f'maskC_epoch{epoch}.pth'))


def overlap_loss(maskA, maskB, maskC):
    # Calculate overlaps between each pair of masks
    overlap_AB = (maskA & maskB).float().sum()
    overlap_AC = (maskA & maskC).float().sum()
    overlap_BC = (maskB & maskC).float().sum()
    
    # Total overlap loss
    loss = overlap_AB + overlap_AC + overlap_BC
    return loss

def train(rank, world_size):
    # already specified in the bash script
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "29502"
    
    torch.cuda.set_device(rank)
    torch.autograd
    # Initialize process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    
    # Model initialization
    if args.flag == 'K1':
        from networks.Network_Stage2_share import UNet
    elif args.flag == 'S1':
        from networks.Network_Stage1 import UNet
    elif args.flag == 'O':
        from networks.Network_our import UNet
    net = UNet(base_channel=base_channel, num_res=num_res)
    # net.log_var_A = nn.Parameter(torch.tensor(0.0))
    print(net.log_var_A)
    # net.log_var_B = nn.Parameter(torch.tensor(0.0))
    # net.log_var_C = nn.Parameter(torch.tensor(0.0))
    net = net.to(rank)
    
    net_eval = UNet(base_channel=base_channel, num_res=num_res)
    pretrained_model = torch.load(args.pre_model, map_location='cpu')
    net.load_state_dict(pretrained_model, strict=False)
    net = DDP(net, device_ids=[rank],find_unused_parameters=True) # TODO ddp_model is name matter
    # net._set_static_graph()   # DDP
    
    # Data loading with DistributedSampler
    train_datasets = get_training_data()
    train_sampler = DistributedSampler(train_datasets, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_datasets, batch_size=args.BATCH_SIZE, num_workers=world_size, sampler=train_sampler)
    
    # Only rank 0 needs to initialize the SummaryWriter and evaluation datasets
    # if rank == 0:
    writer = SummaryWriter(args.writer_dir + exper_name)
    eval_loader_RD = get_eval_data(val_in_path=args.eval_in_path_RD, val_gt_path=args.eval_gt_path_RD)
    eval_loader_Rain = get_eval_data(val_in_path=args.eval_in_path_Rain, val_gt_path=args.eval_gt_path_Rain)
    eval_loader_L = get_eval_data(val_in_path=args.eval_in_path_L, val_gt_path=args.eval_gt_path_L)
    # else:
    #     writer = None
    
    # Optimizer and scheduler
    optimizerG_B1 = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    scheduler_B1 = CosineAnnealingWarmRestarts(optimizerG_B1, T_0=args.T_period, T_mult=1)

    loss_char= losses.CharbonnierLoss()

    vgg = models.vgg16(pretrained=False) # TODO uncomment this line, and change back to False
    vgg.load_state_dict(torch.load('/mnt/pipeline_1/weight/vgg16-397923af.pth'))
    vgg_model = vgg.features[:16]
    vgg_model = vgg_model.to(rank)
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()
    
    step =0
    max_psnr_val_L = args.max_psnr
    max_psnr_val_Rain = args.max_psnr
    max_psnr_val_RD = args.max_psnr

    total_lossA = 0.0
    total_lossB = 0.0
    total_lossC  = 0.0

    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0
    total_loss4 = 0.0
    total_loss5 = 0.0
    total_loss6 = 0.0
    total_loss = 0.0

    input_PSNR_all_A = 0
    train_PSNR_all_A = 0
    input_PSNR_all_B = 0
    train_PSNR_all_B = 0
    input_PSNR_all_C = 0
    train_PSNR_all_C = 0
    Frequncy_eval_save = len(train_loader)
    
    parameter_A = []
    parameter_B = []
    parameter_C = []
    
    maskAs = []
    maskBs = []
    maskCs = []
    
    maskAs_change = []
    maskBs_change = []
    maskCs_change = []   
    
    
    parameters_A=[]
    parameters_B=[]
    parameters_C=[]
    
    


    iter_nums = 0
    # TODO check later
    torch.autograd.set_detect_anomaly(True)
    
    fine_tune = True
    
    if fine_tune:
        original_parameters = [param.clone().detach() for param in net.parameters()]
    
    for epoch in range(args.EPOCH): 
        # train_sampler.set_epoch(epoch)
        scheduler_B1.step(epoch)

        st = time.time()
        # import pdb;pdb.set_trace()
        for idx,train_data in enumerate(train_loader):#   (data_in, label)  ----- train_data
            # net.parameters = original_parameters.clone()
            cloned_parameters = [param.clone() for param in original_parameters]
            with torch.no_grad():
                for param, cloned_param in zip(net.parameters(), cloned_parameters):
                    param.data.copy_(cloned_param.data)

            # net.parameters = [param.clone() for param in original_parameters]
            # print(maskAs)
            # if maskAs is not None:
            #     maskA = maskAs[-1]
            #     maskB = maskBs[-1]
            #     maskC = maskCs[-1]
            # if maskAs and maskBs and maskCs:  # 检查列表是否都不为 None 且不为空
            if maskAs:
                    maskA = maskAs[-1] if maskAs[-1] is not None else torch.zeros_like(cloned_parameters[0], dtype=torch.bool)
            else:
                maskA = torch.zeros_like(cloned_parameters[0], dtype=torch.bool)

            if maskBs:
                maskB = maskBs[-1] if maskBs[-1] is not None else torch.zeros_like(cloned_parameters[1], dtype=torch.bool)
            else:
                maskB = torch.zeros_like(cloned_parameters[1], dtype=torch.bool)

            if maskCs:
                maskC = maskCs[-1] if maskCs[-1] is not None else torch.zeros_like(cloned_parameters[2], dtype=torch.bool)
            else:
                maskC = torch.zeros_like(cloned_parameters[2], dtype=torch.bool)
            # maskA = maskAs[-1]
            # maskB = maskBs[-1]
            # maskC = maskCs[-1]
            # print(maskA)
            
            #data_A, data_B = train_data
            # import pdb;pdb.set_trace()
            data_A, data_B, data_C = train_data
            # if i ==0:
            #     print("data_A.size(),in_GT:",data_A[0].size(), data_A[1].size())  # Snow
            #     print("data_B.size(),in_GT:", data_B[0].size(), data_B[1].size()) # Rain
            #     print("data_C.size(),in_GT:", data_C[0].size(), data_C[1].size()) # RD

            iter_nums = iter_nums + 1
            net.train()


            inputs_A = Variable(data_A[0]).cuda(rank, non_blocking=True)
            labels_A = Variable(data_A[1]).cuda(rank, non_blocking=True)
            inputs_B = Variable(data_B[0]).cuda(rank, non_blocking=True)
            labels_B = Variable(data_B[1]).cuda(rank, non_blocking=True)
            inputs_C = Variable(data_C[0]).cuda(rank, non_blocking=True)
            labels_C = Variable(data_C[1]).cuda(rank, non_blocking=True)
            # print(f"length of dataA is {len(inputs_A)}")
            # print(f"length of dataB is {len(inputs_B)}")
            # print(f"length of dataC is {len(inputs_C)}")
            assert inputs_A.size(0) == inputs_B.size(0) == inputs_C.size(0), "Batch sizes must match"

            # # 沿第一个维度拼接
            # combined_data = torch.cat((inputs_A, inputs_B, inputs_C), dim=0)
            # combined_labels = torch.cat((labels_A, labels_B, labels_C), dim=0)

            # # 计算 1/3 的数量
            # total_length = combined_data.size(0)
            # subset_length = total_length // 3  # 向下取整

            # # 生成随机索引
            # indices = torch.randperm(total_length)[:subset_length]

            # # 根据随机索引提取数据
            # inputs_all = combined_data[indices]
            # labels_all = combined_labels[indices]
            #--------------------------------------------optimizerG_B1---------------------------------------------#

            net.zero_grad()
            optimizerG_B1.zero_grad()

            # ============================== data A  ============================== #
            # net.zero_grad()
            # optimizerG_B1.zero_grad()
            # import pdb;pdb.set_trace()

            if parameter_A:
                parameters_list = list(net.parameters())
                with torch.no_grad():
                    # 遍历 parameters_list 的每个参数并应用 maskA
                    for param, new_val,mask in zip(parameters_list, parameter_A,maskA):
                        param.data[mask] = new_val[mask]  # 仅更新 maskA 为 True 的位置
                # net.parameters()[maskA] = parameter_A
            train_output_A = net(inputs_A)
            input_PSNR_A = compute_psnr(inputs_A, labels_A)
            trian_PSNR_A = compute_psnr(train_output_A, labels_A)
            # import pdb;pdb.set_trace()

            loss1 = F.smooth_l1_loss(train_output_A, labels_A) +  args.VGG_lamda * loss_network(train_output_A, labels_A)
            # print(loss1)
            g_lossA = loss1
            total_lossA += g_lossA.item()
            net.zero_grad()  # 清除梯度
            g_lossA.backward(retain_graph=True)  # 计算梯度
            # gradA = [param.grad.clone() for param in net.parameters() if param.grad is not None]  # Save gradA
            gradA = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) for param in net.parameters()]
            # gradA = [
            #             param.grad.clone() if param.grad is not None else torch.zeros_like(param)
            #             for param in net.parameters()
            #         ]
            
 
            input_PSNR_all_A = input_PSNR_all_A + input_PSNR_A
            train_PSNR_all_A = train_PSNR_all_A + trian_PSNR_A


            # ============================== data B  ============================== #
            # net.zero_grad()
            # optimizerG_B1.zero_grad()
            if parameter_B:
                parameters_list = list(net.parameters())
                with torch.no_grad():
                    # 遍历 parameters_list 的每个参数并应用 maskA
                    for param, new_val,mask in zip(parameters_list, parameter_B,maskB):
                        param.data[mask] = new_val[mask]  # 仅更新 maskA 为 True 的位置
            train_output_B = net(inputs_B)
            input_PSNR_B = compute_psnr(inputs_B, labels_B)
            trian_PSNR_B = compute_psnr(train_output_B, labels_B)

            loss3 = F.smooth_l1_loss(train_output_B, labels_B) + args.VGG_lamda * loss_network(train_output_B, labels_B)

            g_lossB = loss3
            total_lossB += g_lossB.item()
            net.zero_grad()  # 清除梯度
            g_lossB.backward(retain_graph=True)
            # gradB = [param.grad.clone() for param in net.parameters() if param.grad is not None]  # Save gradB
            gradB = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) for param in net.parameters()]
            # gradB = [
            #             param.grad.clone() if param.grad is not None else torch.zeros_like(param)
            #             for param in net.parameters()
            #         ]

            

            input_PSNR_all_B = input_PSNR_all_B + input_PSNR_B
            train_PSNR_all_B = train_PSNR_all_B + trian_PSNR_B

            
            
            # ============================== data C  ============================== #

            if parameter_C:
                parameters_list = list(net.parameters())
                with torch.no_grad():
                    # 遍历 parameters_list 的每个参数并应用 maskA
                    for param, new_val,mask in zip(parameters_list, parameter_C,maskC):
                        param.data[mask] = new_val[mask]  # 仅更新 maskA 为 True 的位置
            train_output_C = net(inputs_C)
            input_PSNR_C = compute_psnr(inputs_C, labels_C)
            trian_PSNR_C = compute_psnr(train_output_C, labels_C)

            loss5 = F.smooth_l1_loss(train_output_C, labels_C) +  args.VGG_lamda * loss_network(train_output_C, labels_C)


            g_lossC =  loss5
            total_lossC += g_lossC.item()
            net.zero_grad()  # 清除梯度
            g_lossC.backward(retain_graph=True)
            # gradC = [param.grad.clone() for param in net.parameters() if param.grad is not None]  # Save gradC
            gradC = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) for param in net.parameters()]

            
            # gradC = [
            #             param.grad.clone() if param.grad is not None else torch.zeros_like(param)
            #             for param in net.parameters()
            #         ]


            input_PSNR_all_C = input_PSNR_all_C + input_PSNR_C
            train_PSNR_all_C = train_PSNR_all_C + trian_PSNR_C

            # g_lossC.backward(retain_graph=True)
            # optimizerG_B1.step()
            
            alphaA = 1/len(inputs_A)
            alphaB = 1/len(inputs_B)
            alphaC = 1/len(inputs_C)
            

            # Define weighting factors for each loss, adjusted by log-variance and data length
            weight_A = alphaA / (2 * torch.exp(net.module.log_var_A)**2)
            weight_B = alphaB / (2 * torch.exp(net.module.log_var_B)**2)
            weight_C = alphaC / (2 * torch.exp(net.module.log_var_C)**2)
            # print(f"grad A is {gradA[10]}")
            
            total_weight = weight_A + weight_B + weight_C

            # 标准化
            weight_A = weight_A / total_weight
            weight_B = weight_B / total_weight
            weight_C = weight_B / total_weight

          
            # TODO updata to traniable para
            # Create masks for each gradient set
            percentage=95
            maskA = calculate_mask(gradA, percentage=percentage) # TODO experiment need change percentage, regarding to the difficulty of the task, check the mask_log
            maskB = calculate_mask(gradB, percentage=percentage)
            maskC = calculate_mask(gradC, percentage=percentage)
            
            # Calculate gradients for the combined loss
            # g_loss = (
            #     (F.smooth_l1_loss(train_output_A, labels_A) +  args.VGG_lamda * loss_network(train_output_A, labels_A)) +
            #     (F.smooth_l1_loss(train_output_B, labels_B) + args.VGG_lamda * loss_network(train_output_B, labels_B)) +
            #     (F.smooth_l1_loss(train_output_C, labels_C) +  args.VGG_lamda * loss_network(train_output_C, labels_C))
            # ) 
            g_loss = (
                weight_A * (F.smooth_l1_loss(train_output_A, labels_A) +  args.VGG_lamda * loss_network(train_output_A, labels_A)) +
                weight_B * (F.smooth_l1_loss(train_output_B, labels_B) + args.VGG_lamda * loss_network(train_output_B, labels_B)) +
                weight_C * (F.smooth_l1_loss(train_output_C, labels_C) +  args.VGG_lamda * loss_network(train_output_C, labels_C))
            ) # + overlap_loss(maskA, maskB, maskC) # TODO experiment need, may restrict the model too much
            # g_loss.backward(retain_graph=False)
            # net.zero_grad()
            # train_output_all = net(inputs_all, flag = [1,0,0])
            # g_loss = F.smooth_l1_loss(train_output_all, labels_all) +  args.VGG_lamda * loss_network(train_output_all, labels_all)
            
            net.zero_grad()
            g_loss.backward(retain_graph=True)
            # grad_total = [param.grad.clone() for param in net.parameters() if param.grad is not None]  # Save grad_total
            grad_total = [param.grad.clone() if param.grad is not None else torch.zeros_like(param) for param in net.parameters()]
            # grad_total = [
            #             param.grad.clone() if param.grad is not None else torch.zeros_like(param)
            #             for param in net.parameters()
            #         ]

          

            # Remove overlaps in masks
            # TODO remove the overlap
            # remove_overlap_masks(maskA, maskB, maskC)
            
            # TODO capture correlation between A and B, A and C, B and C
            
            # Initialize total masks if they are None
            # TODO mask visual of maskA, maskB, maskC
            # Accumulate masks
            maskAs.append(maskA)
            maskBs.append(maskB)
            maskCs.append(maskC)
            
            # 检查每个列表的长度，如果超过 1，就移除第一个元素
            if len(maskAs) > 5:
                maskAs.pop(0)
            if len(maskBs) > 5:
                maskBs.pop(0)
            if len(maskCs) > 5:
                maskCs.pop(0)
                

                
            maskA_change,maskB_change,maskC_change=process_masks(maskAs, maskBs, maskCs)
            # print("******")
            
            maskAs_change.append(maskA_change)
            maskBs_change.append(maskB_change)
            maskCs_change.append(maskC_change)
            
            if len(maskAs_change) > 5:
                maskAs_change.pop(0)
            if len(maskBs_change) > 5:
                maskBs_change.pop(0)
            if len(maskCs_change) > 5:
                maskCs_change.pop(0)
                
                
            
            
            parameters_list = list(net.parameters())
            #遍历parameters_list与mask中的每个变量，将对应位置的参数留下，其他位置用0代替
            
            maskA_paras=[]
            maskB_paras=[]
            maskC_paras=[]
            if maskA_change is not None:
                # parameter_A = net.parameters()[maskA].clone().detach() # FIXME not sure detach is necessary
                  # 将参数生成器转换为列表
                # parameter_A = parameters_list[maskA].clone().detach()  # 现在可以按索引访问
                for param, mask in zip(parameters_list, maskA):  # Assume mask_list is the list of masks with the same structure as parameters_list
                    # print(param.shape)
                    # print(maskA[0].shape)
                    # print(mask)
                    masked_param = param * mask  # Element-wise multiplication, keeping only masked positions
                    maskA_paras.append(masked_param)
                parameters_A.append(maskA_paras)

                if len(parameters_A) > 1:
                    parameters_A.pop(0)
                parameter_A=parameters_A[-1]
                

            if maskB_change is not None:
                for param, mask in zip(parameters_list, maskB):  # Assume mask_list is the list of masks with the same structure as parameters_list
                    masked_param = param * mask  # Element-wise multiplication, keeping only masked positions
                    maskB_paras.append(masked_param)
                
                if len(parameters_B) > 1:
                    parameters_B.pop(0)
                parameters_B.append(maskB_paras)
                parameter_B=parameters_B[-1]
            if maskC_change is not None:
                for param, mask in zip(parameters_list, maskC):  # Assume mask_list is the list of masks with the same structure as parameters_list
                    masked_param = param * mask  # Element-wise multiplication, keeping only masked positions
                    maskC_paras.append(masked_param)
                parameters_C.append(maskC_paras)
                if len(parameters_C) > 1:
                    parameters_C.pop(0)
                parameter_C=parameters_C[-1]

            # Apply masks to grad_total
            for i, param in enumerate(net.parameters()):
                if grad_total[i].sum() == 0:  # 检查grad_total[i]是否全为零
                    grad_total[i] = torch.zeros_like(grad_a)  # 替换为与grad_a相同维度的全零数组
                    
                if param.grad is not None:
                    fine_tune = True  # Define the variable
                    if fine_tune:
                        grad_a= ( 
                            gradA[i] * maskA[i].float()
                        )
                        grad_b= (
                            gradB[i] * maskB[i].float()
                        )
                        grad_c= (
                            gradC[i] * maskC[i].float()
                        )
                        # maskA = torch.tensor(maskA, dtype=torch.bool) if isinstance(maskA, list) else maskA
                        # maskB = torch.tensor(maskB, dtype=torch.bool) if isinstance(maskB, list) else maskB
                        # maskC = torch.tensor(maskC, dtype=torch.bool) if isinstance(maskC, list) else maskC
                        # print(maskA)
                        mask_combined = maskA[i] | maskB[i] | maskC[i]
                        # mask_combined = [a | b | c for a, b, c in zip(maskA, maskB, maskC)]
                        # mask_combined = [a.item() or b.item() or c.item() for a, b, c in zip(maskA, maskB, maskC)]

                        grad_total[i][~mask_combined] = 0
                    else:
                        # print("********")
                        # print(len(list(net.parameters())))
                        # print(len(grad_total))
                        # print(grad_total[i].shape,gradA[i].shape,maskA[i].float().shape)
                        grad_a= ( 
                            0.8 * gradA[i] * maskA[i].float()  +
                            0.1 *  gradB[i] * maskB[i].float()  +
                            0.1 * gradC[i] *maskC[i].float()
                        )
                        grad_b= (
                            0.8 * gradB[i] *maskB[i].float() +
                            0.1 * gradA[i] *maskA[i].float() +
                            0.1 * gradC[i]*maskC[i].float()                         
                        )
                        grad_c= (
                            0.8 * gradC[i]*maskC[i].float() +
                            0.1 * gradA[i]*maskA[i].float() +
                            0.1 * gradB[i]*maskB[i].float() 
                        )


                    # 根据 maskA、maskB 和 maskC 更新 grad_total[i]
                    # import pdb;pdb.set_trace()
                    # print(para_A[i].shape,grad_total[i].shape,maskA[i].shape)
                    if grad_total[i].sum() == 0:  # 检查grad_total[i]是否全为零
                        grad_total[i] = torch.zeros_like(grad_a)  # 替换为与para_A相同维度的全零数组

                    # TODO para
                    grad_total[i][maskA[i]] = grad_a[maskA[i]]
                    grad_total[i][maskB[i]] = grad_b[maskB[i]]
                    grad_total[i][maskC[i]] = grad_c[maskC[i]]
                        
                        
                # for i, param in enumerate(net.parameters()):
                    
                #     grad_total[i][maskA[i]] = 0.8 * weight_A * gradA[i][maskA[i]] + (net.module.log_var_A + net.module.log_var_B + net.module.log_var_C)
                #     grad_total[i][maskB[i]] = 0.8 * weight_B * gradB[i][maskB[i]] + (net.module.log_var_A + net.module.log_var_B + net.module.log_var_C)
                #     grad_total[i][maskC[i]] = 0.8 * weight_C * gradC[i][maskC[i]] + (net.module.log_var_A + net.module.log_var_B + net.module.log_var_C)

                # Update gradients of the network with modified grad_total
            with torch.no_grad():
                for i, param in enumerate(net.parameters()):
                    if param.grad is not None:
                        param.grad = grad_total[i]
                            
                
            optimizerG_B1.step()

            #-----------------------------------------------------------------------------------------#
            total_loss = total_loss + g_loss.item()
            total_loss1 = total_loss1 + loss1.item()
            # total_loss2 += loss2.item()
            total_loss3 = total_loss3 + loss3.item()
            # total_loss4 += loss4.item()
            total_loss5 = total_loss5 + loss5.item()
            # total_loss6 += loss6.item()


            # print(i,(i+1) % args.print_frequency)
            if (idx+1) % args.print_frequency ==0 and idx >1:
                
                print(
                    "[epoch:%d / EPOCH :%d],[%d / %d], [lr: %.7f ],[ weight_A:%.5f,loss1:%.5f, weight_B:%.5f,loss3:%.5f, weight_C:%.5f,loss5:%.5f, avg_lossA:%.5f, avg_lossB:%.5f, avg_lossC:%.5f, avg_loss:%.5f],"
                    "[in_PSNR_A: %.3f, out_PSNR_A: %.3f],[in_PSNR_B: %.3f, out_PSNR_B: %.3f],[in_PSNR_C: %.3f, out_PSNR_C: %.3f],"
                    "time: %.3f" %
                    (epoch,args.EPOCH, i + 1, len(train_loader), optimizerG_B1.param_groups[0]["lr"], weight_A.item(),loss1.item(),
                     weight_B.item(),loss3.item(),weight_C.item(), loss5.item(),total_lossA / iter_nums,total_lossB / iter_nums, total_lossC / iter_nums,total_loss / iter_nums,
                     input_PSNR_A, trian_PSNR_A, input_PSNR_B, trian_PSNR_B, input_PSNR_C, trian_PSNR_C,time.time() - st))

                st = time.time()
            # if args.SAVE_Inter_Results:
            #     save_path = SAVE_Inter_Results_PATH + str(iter_nums) + '.jpg'
            #     save_imgs_for_visual(save_path, inputs, labels, train_output)
        
        # Save accumulated masks to local directory at the end of the epoch
        #TODO mask visual
        save_masks_to_local(maskAs, maskBs, maskCs, epoch)
        save_masks_to_local(maskAs_change, maskBs_change, maskCs_change, epoch)
        #TODO save three different models
        # save_model_A = save_masked_model(net, maskA, original_parameters, epoch, 'snow')
        # save_model_B = save_masked_model(net, maskB, original_parameters, epoch, 'rain')
        # save_model_C = save_masked_model(net, maskC, original_parameters, epoch, 'raindrop')
        
        
        
        save_model = SAVE_PATH  + 'net_epoch_{}.pth'.format(epoch)
        torch.save(net.module.state_dict(),save_model)
        
        
        max_psnr_val_L = test(net= net_eval, save_model = save_model,  eval_loader = eval_loader_L,epoch=epoch,max_psnr_val = max_psnr_val_L, Dname = 'Snow-L',flag = [1,0,0])
        max_psnr_val_Rain = test(net=net_eval, save_model = save_model, eval_loader = eval_loader_Rain, epoch=epoch, max_psnr_val=max_psnr_val_Rain, Dname= 'HRain',flag = [0,1,0])
        max_psnr_val_RD = test(net=net_eval, save_model  = save_model, eval_loader = eval_loader_RD, epoch=epoch, max_psnr_val=max_psnr_val_RD, Dname= 'RD',flag = [0,0,1] )
    

        # max_psnr_val_L = test(net= net_eval, save_model = save_model_A,  eval_loader = eval_loader_L,epoch=epoch,max_psnr_val = max_psnr_val_L, Dname = 'Snow-L',flag = [1,0,0])
        # max_psnr_val_Rain = test(net=net_eval, save_model = save_model_B, eval_loader = eval_loader_Rain, epoch=epoch, max_psnr_val=max_psnr_val_Rain, Dname= 'HRain',flag = [0,1,0])
        # max_psnr_val_RD = test(net=net_eval, save_model  = save_model_C, eval_loader = eval_loader_RD, epoch=epoch, max_psnr_val=max_psnr_val_RD, Dname= 'RD',flag = [0,0,1] )
    # TODO save the mask
def main():
    try:
        # Set the multiprocessing start method to 'fork', which is often required for PyTorch's multiprocessing.
        mp.set_start_method('fork', force=True)

        # Spawn multiple processes to run the training function in parallel, based on the defined world size.
        # `args.world_size` specifies the number of processes, and each process calls `train`.
        mp.spawn(train,
                 args=(args.world_size,),
                 nprocs=args.world_size,
                 join=True)

        # Synchronize all processes to ensure they have completed execution.
        torch.distributed.barrier() 

    except Exception as ex:
        # Catch and print any errors that occur during process spawning or synchronization.
        print(f"An error occurred: {ex}")
 


if __name__ == '__main__':
    main()