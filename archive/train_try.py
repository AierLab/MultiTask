import os,cv2,time,torchvision,argparse,logging,sys,os,gc
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
parser.add_argument('--experiment_name', type=str,default= "training_try_stage2_share") # modify the experiments name-->modify all save path
parser.add_argument('--unified_path', type=str,default=  '/mnt/pipeline_1/MLT/Weather/')
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
parser.add_argument('--pre_model', type=str,default= '/mnt/pipeline_1/MLT/Weather/training_stage1/net_epoch_99.pth')

#training setting
parser.add_argument('--base_channel', type = int, default= 20)
parser.add_argument('--num_block', type=int, default= 6)
parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
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


def test(net,eval_loader, save_model ,epoch =1,max_psnr_val=26 ,Dname = 'S',flag = [1,0,0]):
    net.to('cuda:0')
    net.eval()
    net.load_state_dict(torch.load(save_model), strict=True)

    st = time.time()
    with torch.no_grad():
        eval_output_psnr = 0.0
        eval_input_psnr = 0.0
        for index, (data_in, label, name) in enumerate(eval_loader, 0):#enumerate(tqdm(eval_loader), 0):
            inputs = Variable(data_in).to('cuda:0')
            labels = Variable(label).to('cuda:0')

            outputs = net(inputs,flag=flag)
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


def example(rank, world_size=4):
    torch.autograd
    world_size=4

    # Initialize process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    
    # Model initialization
    if args.flag == 'K1':
        from networks.Network_Stage2_share import UNet
    elif args.flag == 'K3':
        from networks.Network_Stage2_K3_Flag import UNet
    net = UNet(base_channel=base_channel, num_res=num_res).to(rank)
    net_eval = UNet(base_channel=base_channel, num_res=num_res)
    # pretrained_model = torch.load(args.pre_model, map_location='cpu')
    # net.load_state_dict(pretrained_model, strict=False)
    net = DDP(net, device_ids=[rank]) # TODO ddp_model is name matter
    
    # Data loading with DistributedSampler
    train_datasets = get_training_data()
    train_sampler = DistributedSampler(train_datasets, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_datasets, batch_size=args.BATCH_SIZE, num_workers=0, sampler=train_sampler)
    
    # Only rank 0 needs to initialize the SummaryWriter and evaluation datasets
    if rank == 0:
        writer = SummaryWriter(args.writer_dir + exper_name)
        eval_loader_RD = get_eval_data(val_in_path=args.eval_in_path_RD, val_gt_path=args.eval_gt_path_RD)
        eval_loader_Rain = get_eval_data(val_in_path=args.eval_in_path_Rain, val_gt_path=args.eval_gt_path_Rain)
        eval_loader_L = get_eval_data(val_in_path=args.eval_in_path_L, val_gt_path=args.eval_gt_path_L)
    else:
        writer = None
    
    # Optimizer and scheduler
    optimizerG_B1 = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    scheduler_B1 = CosineAnnealingWarmRestarts(optimizerG_B1, T_0=args.T_period, T_mult=1)

    loss_char= losses.CharbonnierLoss()

    vgg = models.vgg16(pretrained=False) # TODO uncomment this line, and change back to False
    vgg.load_state_dict(torch.load('/mnt/pipeline_1/weight/vgg16-397923af.pth'))
    vgg_model = vgg.features[:16]
    vgg_model = vgg_model.to(rank)
    # for param in vgg_model.parameters():
    #     param.requires_grad = False
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

    iter_nums = 0
    # TODO check later
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.EPOCH):
        # train_sampler.set_epoch(epoch)
        scheduler_B1.step(epoch)

        st = time.time()
        # import pdb;pdb.set_trace()
        for i,train_data in enumerate(train_loader):#   (data_in, label)  ----- train_data
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
            #--------------------------------------------optimizerG_B1---------------------------------------------#


            # ============================== data A  ============================== #
            net.zero_grad()
            optimizerG_B1.zero_grad()
            # import pdb;pdb.set_trace()
            train_output_A = net(inputs_A, flag = [1,0,0])
            input_PSNR_A = compute_psnr(inputs_A, labels_A)
            trian_PSNR_A = compute_psnr(train_output_A, labels_A)
            # import pdb;pdb.set_trace()

            loss1 = F.smooth_l1_loss(train_output_A, labels_A) +  args.VGG_lamda * loss_network(train_output_A.clone(), labels_A.clone())
            
            g_lossA = loss1
            total_lossA = total_lossA + g_lossA.item()
            input_PSNR_all_A = input_PSNR_all_A + input_PSNR_A
            train_PSNR_all_A = train_PSNR_all_A + trian_PSNR_A
            print(f"***********loss is {g_lossA}")
            optimizerG_B1.zero_grad()
            g_lossA.backward(retain_graph=False)
            optimizerG_B1.step()
            print("A is ok")

            # ============================== data B  ============================== #
            net.zero_grad()
            optimizerG_B1.zero_grad()

            train_output_B = net(inputs_B, flag = [0,1,0])
            input_PSNR_B = compute_psnr(inputs_B, labels_B)
            trian_PSNR_B = compute_psnr(train_output_B, labels_B)

            loss3 = F.smooth_l1_loss(train_output_B, labels_B) + args.VGG_lamda * loss_network(train_output_B, labels_B)

            g_lossB = loss3
            total_lossB = total_lossB + g_lossB.item()

            input_PSNR_all_B = input_PSNR_all_B + input_PSNR_B
            train_PSNR_all_B = train_PSNR_all_B + trian_PSNR_B


            train_output_C = net(inputs_C,flag = [0, 0, 1])
            input_PSNR_C = compute_psnr(inputs_C, labels_C)
            trian_PSNR_C = compute_psnr(train_output_C, labels_C)

            loss5 = F.smooth_l1_loss(train_output_C, labels_C) +  args.VGG_lamda * loss_network(train_output_C, labels_C)


            g_lossC =  loss5
            total_lossC  = total_lossC +  g_lossC.item()
            input_PSNR_all_C = input_PSNR_all_C + input_PSNR_C
            train_PSNR_all_C = train_PSNR_all_C + trian_PSNR_C



            g_loss = g_lossA + g_lossB + g_lossC

            #-----------------------------------------------------------------------------------------#
            total_loss = total_loss + g_loss.item()
            total_loss1 = total_loss1 + loss1.item()
            # total_loss2 += loss2.item()
            total_loss3 = total_loss3 + loss3.item()
            # total_loss4 += loss4.item()
            total_loss5 = total_loss5 + loss5.item()
            # total_loss6 += loss6.item()



            if (i+1) % args.print_frequency ==0 and i >1:
                print(
                    "[epoch:%d / EPOCH :%d],[%d / %d], [lr: %.7f ],[loss1:%.5f,loss3:%.5f,loss5:%.5f, avg_lossA:%.5f, avg_lossB:%.5f, avg_lossC:%.5f, avg_loss:%.5f],"
                    "[in_PSNR_A: %.3f, out_PSNR_A: %.3f],[in_PSNR_B: %.3f, out_PSNR_B: %.3f],[in_PSNR_C: %.3f, out_PSNR_C: %.3f],"
                    "time: %.3f" %
                    (epoch,args.EPOCH, i + 1, len(train_loader), optimizerG_B1.module.param_groups[0]["lr"],  loss1.item(),
                     loss3.item(), loss5.item(),total_lossA / iter_nums,total_lossB / iter_nums, total_lossC / iter_nums,total_loss / iter_nums,
                     input_PSNR_A, trian_PSNR_A, input_PSNR_B, trian_PSNR_B, input_PSNR_C, trian_PSNR_C,time.time() - st))

                st = time.time()
            # if args.SAVE_Inter_Results:
            #     save_path = SAVE_Inter_Results_PATH + str(iter_nums) + '.jpg'
            #     save_imgs_for_visual(save_path, inputs, labels, train_output)
        save_model = SAVE_PATH  + 'net_epoch_{}.pth'.format(epoch)
        torch.save(net.module.state_dict(),save_model)

        max_psnr_val_L = test(net= net_eval, save_model = save_model,  eval_loader = eval_loader_L,epoch=epoch,max_psnr_val = max_psnr_val_L, Dname = 'Snow-L',flag = [1,0,0])
        max_psnr_val_Rain = test(net=net_eval, save_model = save_model, eval_loader = eval_loader_Rain, epoch=epoch, max_psnr_val=max_psnr_val_Rain, Dname= 'HRain',flag = [0,1,0])
        max_psnr_val_RD = test(net=net_eval, save_model  = save_model, eval_loader = eval_loader_RD, epoch=epoch, max_psnr_val=max_psnr_val_RD, Dname= 'RD',flag = [0,0,1] )
    
def main():
    try:
        mp.spawn(example,
                args=(args.rank,),
                nprocs=4,
                join=True)
    except Exception as ex:
        print(f"An error occurred: {ex}")     


if __name__ == '__main__':
    # import os
    os.environ['MASTER_PORT'] = '29500'  # 选择一个未被占用的端口号

    os.environ["NCCL_DEBUG"] = "INFO"
    
    main()