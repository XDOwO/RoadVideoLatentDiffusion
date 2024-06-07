#train.py
from dataset import DiffusionDataset
from diffusion import Diffusion
from unet import Unet
from ae import Encoder,Decoder,CarEncoder,CarDecoder,Mixer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import torch.optim as optim
import os
from torchvision import transforms
import torch.nn.functional as F
def weighted_mse(targets,predicts,xys,xylens,full_weight=1000,mode="car"):
    assert targets.shape[0]==xys.shape[0], "target and xy batch should be the same."
    mask = torch.full_like(targets,1) if mode == "road" else torch.full_like(targets, -1)
    w,h = targets.shape[2:]
    for b in range(targets.shape[0]):
      for xy_n in range(xys.shape[1]):
        x,y = xys[b][xy_n]
        x_rad, y_rad = xylens[b][xy_n]
        if x==y==0:
          continue
        mask[b][:,max(0,y-y_rad//2):min(y+y_rad//2,h):,max(x-x_rad//2,0):min(x+x_rad//2,w)] = -1 if mode == "road" else 1
    
    # t2 =transforms.Compose([
    #     transforms.Lambda(lambda t: (t + 1) / 2),
    #     transforms.ToPILImage(),
    # ])
    # print("xys",xys[0][0],"xylens",xylens[0][0])
    clone_t = targets.clone()
    clone_p = predicts.clone()
    clone_t[mask == -1] = -1
    clone_p[mask == -1] = -1
    # t2(clone_t[0]).save("/home/fish/OursLatentDiffusion/target.png")
    # t2(mask[0]).save("/home/fish/OursLatentDiffusion/mask.png")
    # print("saved")
    # assert 1==0
        
    # xt,_ = self.forward_process(gt,ts,targets)
    # xt_minus_1,_ = self.forward_process(gt,torch.clamp(ts,min=0),noise=targets)
    # predicts = self.backward_step(xt,ts,eps = predicts)
    return torch.mean(((clone_t-clone_p)**2)*full_weight)

def add_car(emp,car,xy,xylen):
    if(emp.shape == 4):
        assert NotImplementedError
    for i in range(xylen.shape[0]):
        xlen,ylen = xylen[i]
        if not xlen or not ylen:
           continue
        x,y = xy[i]
        new_car = F.interpolate(car[i].unsqueeze(0), size=tuple((ylen,xlen)), mode='bilinear', align_corners=False)
        y_odd = int(ylen % 2 == 1)
        x_odd = int(xlen % 2 == 1)
        # Sometimes the shape doens't match, so just let it go
        try:
            emp[:,y-ylen//2:y+ylen//2+y_odd,x-xlen//2:x+xlen//2+x_odd] = new_car
        except:
           pass
       
if __name__ == "__main__":
    os.makedirs(os.path.join("/mnt/4TB/fish/miles_mixer/"),exist_ok=True)
    os.makedirs(os.path.join("./test_result"),exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',"--dataset_path",default="../OursDiffusion/dataset",help="Path to dataset")
    parser.add_argument('-s',"--save_path",default='/mnt/4TB/fish/miles_mixer/',help="Path to store models and test pictures")
    parser.add_argument("-r","--ratio",type=int,default=1,help="Loss ratio of road and cars, if ratio > 1, then model will tend to generate road more.")
    parser.add_argument("-d","--device",type=int,default=0,help="Index of GPU")
    parser.add_argument("-l","--lr",type=float,default=1e-3,help="learning rate")
    
    args = parser.parse_args()

    save_path = args.save_path
    ratio = args.ratio
    dataset_path = args.dataset_path
    device = "cuda:"+str(args.device)
    lr = args.lr
    dataset = DiffusionDataset(dataset_path)
    print("dataset length:",len(dataset))
    print("GPU in use:",device)
    dataloader= DataLoader(dataset,batch_size=128,shuffle = True)
    test_dataloader = DataLoader(dataset,batch_size=10,shuffle = True)
    encoder,decoder = Encoder().to(device),Decoder().to(device)
    c_encoder = CarEncoder().to(device)
    c_decoder = CarDecoder().to(device)
    mixer = Mixer().to(device)
    # encoder.load_state_dict(torch.load("/mnt/4TB/fish/weight_sheep_AE_8192_OURcars/Enc_1250.pth"))
    # decoder.load_state_dict(torch.load("/mnt/4TB/fish/weight_sheep_AE_8192_OURcars/Dec_1250.pth"))
    # c_encoder.load_state_dict(torch.load("/mnt/4TB/fish/Sheep_AE_32x32_0323/Encoder32x32_7_DinJi.pth"))
    epoch = 1000
    loss = nn.MSELoss()
    opt = optim.Adam(list(c_decoder.parameters())+list(c_encoder.parameters()),lr = lr)
    noise = None

    for E in range(epoch):
        # mixer.train()
        # encoder.train()
        # decoder.train()
        c_encoder.train()
        c_decoder.train()


        all_loss_r = 0
        all_loss_c = 0
        all_loss = 0
        loop = tqdm(dataloader,desc="Training", position=0, leave=True)
        for gt,car,emp,xys,xylens,rgbs in loop:
            # t = torch.randint(0,T,size=(gt.shape[0],),dtype=torch.long)
            gt, car, emp, xys, xylens, rgbs = gt.to(device), car.to(device), emp.to(device), xys.to(device), xylens.to(device) , rgbs.to(device)
            
            B,N,C,H,W = car.shape
            output = c_decoder(c_encoder(car,xys,xylens,rgbs))
            # Calculate car loss first
            l = loss(output,car.view(B*N,3,64,64))
            l.backward()
            all_loss+=l.cpu().item()
            opt.step()
            opt.zero_grad()
            loop.set_description(f'Epoch [{E}/{epoch}]')
            loop.set_postfix(loss=l.cpu().item())
        print("[!]Epoch:{},loss:{}".format(E,all_loss/len(dataloader)))
        if (E) % 5 == 0:
            # mixer.eval()
            # encoder.eval()
            # decoder.eval()
            c_encoder.eval()
            c_decoder.eval()
            # torch.save(mixer.state_dict(),save_path+"Mixer_{}.pth".format(E))
            torch.save(c_encoder.state_dict(),save_path+"CEncoder_{}.pth".format(E))
            torch.save(c_decoder.state_dict(),save_path+"CDecoder_{}.pth".format(E))
            # torch.save(encoder.state_dict(),save_path+"Encoder_{}.pth".format(E))
            # torch.save(decoder.state_dict(),save_path+"Decoder_{}.pth".format(E))

            gt, car, emp, xys, xylens, rgbs = next(iter(test_dataloader))
            gt, car, emp, xys, xylens, rgbs = gt.to(device), car.to(device), emp.to(device), xys.to(device), xylens.to(device) , rgbs.to(device)
            B,N,C,H,W = car.shape
            car = c_decoder(c_encoder(car,xys,xylens,rgbs)).view(B,N,3,64,64)
            add_car(emp[0],car[0],xys[0],xylens[0])
            t = transforms.Compose([
                transforms.Lambda(lambda t: (t + 1) / 2),
                transforms.Resize((128,128),antialias=True),
                transforms.ToPILImage(),
            ])
            t(gt[0]).save("./test_result/gt_{}.png".format(E))
            t(emp[0]).save("./test_result/res_{}.png".format(E))
            print("Saved!")

       