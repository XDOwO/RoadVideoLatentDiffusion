#train.py
import torch.utils
from dataset import DiffusionDataset
from diffusion import Diffusion
from unet import Unet
from ae import CarEncoder,CarDecoder,Encoder,Decoder
from torch.utils.data import DataLoader
import torch
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',"--dataset_path",default="../OursDiffusion/dataset",help="Path to dataset")
    parser.add_argument('-s',"--save_path",default='./save',help="Path to store models and test pictures")
    parser.add_argument("-r","--ratio",type=int,default=1,help="Loss ratio of road and cars, if ratio > 1, then model will tend to generate road more.")
    parser.add_argument("-d","--device",type=int,default=0,help="Index of GPU")
    parser.add_argument("-l","--lr",type=float,default=5e-5,help="learning rate")
    
    args = parser.parse_args()

    save_path = args.save_path
    ratio = args.ratio
    dataset_path = args.dataset_path
    device = "cuda:"+str(args.device)
    lr = args.lr
    dataset = DiffusionDataset("/home/fish/CSE_AILab/milesAE/dataset_extended/dataset_extened_gushan/")
    dataset2 = DiffusionDataset("/home/fish/CSE_AILab/milesAE/dataset_extended/dataset_extended_gushan0601/")
    dataset = torch.utils.data.ConcatDataset([dataset, dataset2])
    train_dataset,test_dataset = torch.utils.data.random_split(dataset,[int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)])
    print("dataset length:",len(dataset))
    print("GPU in use:",device)
    dataloader=DataLoader(train_dataset,batch_size=64,shuffle = True,num_workers=20)
    test_dataloader = DataLoader(test_dataset,batch_size=64,shuffle = True,num_workers=20)
    #original ch_mult:2,4,8 0.0337
    unet = Unet(in_ch=128,model_ch=256,out_ch=128,num_res=4,time_emb_mult=6,dropout=0,ch_mult = (2,4,6), num_heads=8, attention_at_height = 0).to(device)
    encoder,decoder,c_encoder,c_decoder = Encoder().to(device),Decoder().to(device),CarEncoder().to(device),CarDecoder().to(device)
    encoder.load_state_dict(torch.load("/home/fish/OursLatentDiffusion/save/model/miles_mixer/Enc_90.pth"))
    c_encoder.load_state_dict(torch.load("/home/fish/OursLatentDiffusion/save/model/miles_mixer/CEncoder_70.pth"))
    c_decoder.load_state_dict(torch.load("/home/fish/OursLatentDiffusion/save/model/miles_mixer/CDecoder_70.pth"))
    unet.load_state_dict(torch.load("/home/fish/OursLatentDiffusion/save/model/OurDiffusion_2dataset_1.pth"))
    diffusion_model = Diffusion(1000,unet,encoder,decoder,c_encoder,c_decoder,save_path,"cosine",device = device)
    diffusion_model.train_model(100000,dataloader,test_dataloader,lr = lr,ratio = ratio,save_interval = 5)
