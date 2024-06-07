#test.py
from dataset import DiffusionDataset
from diffusion import Diffusion,add_car
from unet import Unet
from ae import CarEncoder,CarDecoder,Encoder,Decoder
from xy_predictor import XYLenPredictor
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from tqdm import tqdm
import os
save_path = "./test_result/test"
device = "cuda:1"
if __name__ == "__main__":
    dataset = DiffusionDataset("/home/fish/CSE_AILab/milesAE/dataset_extended/dataset_extened_gushan/")
    # dataset2 = DiffusionDataset("../OursDiffusion/dataset_extened")
    # dataset = torch.utils.data.ConcatDataset([dataset, dataset2])
    dataloader=DataLoader(dataset,batch_size=1,shuffle=True)
    print("dataset length:",len(dataset))
    print("GPU in use:",device)
   
    unet = Unet(in_ch=128,model_ch=256,out_ch=128,num_res=4,time_emb_mult=6,dropout=0,ch_mult = (2,4,6), num_heads=8, attention_at_height = 0).to(device)
    encoder,decoder,c_encoder,c_decoder,xylen_predictor = Encoder().to(device),Decoder().to(device),CarEncoder().to(device),CarDecoder().to(device),XYLenPredictor().to(device)
    encoder.load_state_dict(torch.load("/home/fish/OursLatentDiffusion/save/model/miles_mixer/Enc_90.pth"))
    decoder.load_state_dict(torch.load("/home/fish/OursLatentDiffusion/save/model/miles_mixer/Dec_90.pth"))
    c_encoder.load_state_dict(torch.load("/home/fish/OursLatentDiffusion/save/model/miles_mixer/CEncoder_70.pth"))
    c_decoder.load_state_dict(torch.load("/home/fish/OursLatentDiffusion/save/model/miles_mixer/CDecoder_70.pth"))
    xylen_predictor.load_state_dict(torch.load("/home/fish/OursLatentDiffusion/save/model/miles_mixer/XYLenPredictor_40.pth"))
    unet.load_state_dict(torch.load("/home/fish/OursLatentDiffusion/save/model/OurDiffusion_2dataset_27.pth"))
    diffusion_model = Diffusion(1000,unet,encoder,decoder,c_encoder,c_decoder,save_path,"cosine",device = device)
    unet.eval()
    encoder.eval()
    decoder.eval()
    c_encoder.eval()
    c_decoder.eval()
    loop = tqdm(dataloader,desc="Training", position=0, leave=True)
    i = 0
    for gt,car,emp,xy,xylen,rgbs in loop:
      
      emp_static = emp.clone().to(device)
      gt, emp, xy, car, xylen, rgbs = gt.to(device), emp.to(device), xy.to(device), car.to(device), xylen.to(device) , rgbs.to(device)
      gt = torch.repeat_interleave(gt,5,0)
      emp = torch.repeat_interleave(emp,5,0)
      xy = torch.repeat_interleave(xy,5,0)
      car = torch.repeat_interleave(car,5,0)
      xylen = torch.repeat_interleave(xylen,5,0)
      rgbs = torch.repeat_interleave(rgbs,5,0)
      # print(xy)
      ### 隨機亂生xy
      indices_forever = xy[:,:,0].nonzero()[:,1]
      # print(indices)
      xy_list = [
        [110,55],
        [103,47],
        [94,40],
        [87,34],
        [80,28]
      ]
      xy[:,0] = torch.tensor(xy_list,dtype=torch.int,device=xy.device)
      car[:,1:,:,:,:]=0
      xy[:,1:,:]=0
      xylen[:,1:,:]=0
      rgbs[:,1:,:]=0
      # random_values = torch.randint(0, 128, (len(non_zero_indices[0]),), dtype=torch.int32, device=xy.device)
      
      ### 到此為止
      print(xy)
    
      emp_e = encoder(emp,xy,xylen,rgbs)
  
      B,N,C,H,W = car.shape
      ncar = car.view(B*N,C,H,W)
      B,N,C = xy.shape
      nxy = xy.view(B*N,C)
      B,N,C = xylen.shape
      nxylen = xylen.view(B*N,C)
      B,N,C = rgbs.shape
      nrgbs = rgbs.view(B*N,C)
      
      indices = torch.nonzero(torch.sum(nxylen,dim=1),as_tuple=True)
      repeat = torch.histc(indices[0],bins=emp_e.shape[0],min=0,max=16*emp_e.shape[0])
      

      rds = emp_e.repeat_interleave(repeat,dim=0)
      nxy = nxy[indices]
      nnxylen = torch.tensor((xylen_predictor(nxy,rds) + 1)*64,dtype=torch.int)
      print(nxy.shape,nnxylen.shape)
      # nxy //= 64
      nxylen = nxylen[indices]
      print("nnxylen",nnxylen)
      print("nxylen",nxylen)
      nrgbs = nrgbs[indices] #// 32
      ncar = ncar[indices]
      nrgbs[len(xy_list):]=0
      encoded_car = c_encoder(ncar,nxy,nnxylen,nrgbs)
      nxylen[:]=0
      encoded_car[len(xy_list):] = 0
      sample = diffusion_model.backward_process(encoded_car.shape,emp_e,encoded_car,nxy,nxylen,nrgbs,indices)
      dcar = c_decoder(sample).view(*ncar.shape)
      t = transforms.Compose([
                transforms.Lambda(lambda t: (t + 1) / 2),
                transforms.ToPILImage(),
        ])
      ims = []
      for j in range(len(xy_list)):
        xy[j][1:]=0
        nnxylen = torch.tensor((xylen_predictor(nxy,rds) + 1)*64,dtype=torch.int)
        # xylen[0][1:]=0
        one_nxylen = nnxylen[j].unsqueeze(0)
        one_nxylen[1:] = 0
        add_car(emp[j],dcar[j].unsqueeze(0),xy[j],one_nxylen)
        encoded = encoder(emp,xy,xylen,rgbs)
        decoded = decoder(encoded)
        im = torch.cat([gt[j],emp[j],decoded[j]],dim=-1)
        t(im).save("./test_result/test/res_{}_{}.png".format(i,j))
        t(emp[j]).save(f"./test_result/test/WFD{j}.png")
        ims.append(t(im))
      ims[0].save(f'./test_result/test/res_{i}.gif',save_all=True,append_images=ims[1:],duration=500,loop=0)
      i+=1

      

