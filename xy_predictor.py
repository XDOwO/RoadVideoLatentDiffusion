#train.py
from dataset import DiffusionDataset
from ae import Encoder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import torch.optim as optim
import os
from torchvision import transforms
import torch.nn.functional as F
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn, MofNCompleteColumn,TextColumn,track

class ResBlock(nn.Module):
  def __init__(self,in_ch,emb_ch,out_ch,dropout=0.2):
    super().__init__()
    self.input_layer = nn.Sequential(
        nn.GroupNorm(num_groups=min(in_ch,32), num_channels=in_ch),
        nn.SiLU(),
        nn.Conv1d(in_ch,out_ch,kernel_size=3,padding=1,stride=1))
    self.emb_layer = nn.Sequential(
        nn.SiLU(),
        nn.Linear(emb_ch,out_ch),
    )
    self.output_layer = nn.Sequential(
        nn.GroupNorm(num_groups=min(out_ch,32), num_channels=out_ch),
        nn.SiLU(),
        nn.Dropout(p=dropout),
        nn.Conv1d(out_ch,out_ch,kernel_size=3,padding=1,stride=1),
    )
    self.skip_connection = nn.Conv1d(in_ch,out_ch,kernel_size=3,padding=1,stride=1)
  def forward(self,x,emb):
    xx = self.input_layer(x)
    emb = self.emb_layer(emb)
    while len(emb.shape) < len(xx.shape):
      emb=emb[...,None]
    xx = xx + emb
    xx = self.output_layer(xx)
    return self.skip_connection(x) + xx
class XYLenPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(128,4096)
        self.res = ResBlock(4096,8192,1024)
        self.res2 = ResBlock(1024,8192,32)
        self.res3 = ResBlock(32,8192,1)
        self.acti = nn.Tanh()
    def forward(self,xy,rds):
       emb = self.emb(xy).view(xy.shape[0],-1)
       nrds = rds.view(emb.shape[0],-1,2)
       return self.acti(self.res3(self.res2(self.res(nrds,emb),emb),emb)).view(emb.shape[0],-1)

class ResBlock(nn.Module):
  def __init__(self,in_ch,emb_ch,out_ch,dropout=0.2):
    super().__init__()
    self.input_layer = nn.Sequential(
        nn.GroupNorm(num_groups=min(in_ch,32), num_channels=in_ch),
        nn.SiLU(),
        nn.Conv1d(in_ch,out_ch,kernel_size=3,padding=1,stride=1))
    self.emb_layer = nn.Sequential(
        nn.SiLU(),
        nn.Linear(emb_ch,out_ch),
    )
    self.output_layer = nn.Sequential(
        nn.GroupNorm(num_groups=min(out_ch,32), num_channels=out_ch),
        nn.SiLU(),
        nn.Dropout(p=dropout),
        nn.Conv1d(out_ch,out_ch,kernel_size=3,padding=1,stride=1),
    )
    self.skip_connection = nn.Conv1d(in_ch,out_ch,kernel_size=3,padding=1,stride=1)
  def forward(self,x,emb):
    xx = self.input_layer(x)
    emb = self.emb_layer(emb)
    while len(emb.shape) < len(xx.shape):
      emb=emb[...,None]
    xx = xx + emb
    xx = self.output_layer(xx)
    return self.skip_connection(x) + xx
if __name__ == "__main__":
    os.makedirs(os.path.join("/mnt/4TB/fish/miles_mixer/"),exist_ok=True)
    os.makedirs(os.path.join("./test_result"),exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',"--dataset_path",default="../OursDiffusion/dataset",help="Path to dataset")
    parser.add_argument('-s',"--save_path",default='/mnt/4TB/fish/miles_mixer/',help="Path to store models and test pictures")
    parser.add_argument("-r","--ratio",type=int,default=1,help="Loss ratio of road and cars, if ratio > 1, then model will tend to generate road more.")
    parser.add_argument("-d","--device",type=int,default=1,help="Index of GPU")
    parser.add_argument("-l","--lr",type=float,default=1e-5,help="learning rate")
    
    args = parser.parse_args()

    save_path = args.save_path
    ratio = args.ratio
    dataset_path = args.dataset_path
    device = "cuda:"+str(args.device)
    lr = args.lr
    dataset = DiffusionDataset(dataset_path)
    dataset2 = DiffusionDataset("../OursDiffusion/dataset_tainan")
    dataset = torch.utils.data.ConcatDataset([dataset, dataset2])

    print("dataset length:",len(dataset))
    print("GPU in use:",device)
    dataloader= DataLoader(dataset,batch_size=1024,shuffle = True,num_workers=32)
    test_dataloader = DataLoader(dataset,batch_size=10,shuffle = True)
    model = XYLenPredictor().to(device)
    # c_encoder.load_state_dict(torch.load("/mnt/4TB/fish/Sheep_AE_32x32_0323/Encoder32x32_7_DinJi.pth"))
    epoch = 1000
    loss = nn.MSELoss()
    opt = optim.Adam(model.parameters(),lr = lr)
    noise = None
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load("/mnt/4TB/fish/weight_fish_ae_labeled_128/Enc_500.pth"))
    for E in range(epoch):
        

        all_loss_r = 0
        all_loss_c = 0
        all_loss = 0
        with Progress(
        "[progress.description]{task.description}",  # 顯示任務描述
        TaskProgressColumn(),
        BarColumn(bar_width=None),  # 顯示進度條
        MofNCompleteColumn(),  # 顯示進度百分比
        TextColumn("["),
        TimeElapsedColumn(),  # 顯示已經過時間
        TextColumn("<"),
        TimeRemainingColumn(compact=True),  # 顯示剩餘時間
        TextColumn(",[orange1]{task.fields[spd]}it/s]"),
        TextColumn("[red][bold]Loss:{task.fields[loss]}"),
        transient=True  # 任務完成後自動隱藏進度條
    ) as progress:
            task = progress.add_task(f"[blue]Epoch:{{{E}/{epoch}}}",total=len(dataloader),loss=0,spd=0)
            
            for gt,car,emp,xys,xylens,rgbs in dataloader:
                model.train()

                # t = torch.randint(0,T,size=(gt.shape[0],),dtype=torch.long)
                gt, car, emp, xys, xylens, rgbs = gt.to(device), car.to(device), emp.to(device), xys.to(device), xylens.to(device) , rgbs.to(device)
                emp_e = encoder(emp,xys,xylens,rgbs)

                B,N,C = xys.shape
                nxy = xys.view(B*N,C)
                B,N,C = xylens.shape
                nxylen = xylens.view(B*N,C) 
                indices = torch.nonzero(torch.sum(nxylen,dim=1),as_tuple=True)
                nxy = nxy[indices]
                nxylen = nxylen[indices] / 127 * 2 - 1
                repeat = torch.histc(indices[0],bins=emp_e.shape[0],min=0,max=16*emp_e.shape[0])
                rds = emp_e.repeat_interleave(repeat,dim=0)
                output = model(nxy,rds)
                # Calculate car loss first
                
                l = loss(output,nxylen)
                l.backward()
                all_loss+=l.cpu().item()
                opt.step()
                opt.zero_grad()
                progress.tasks[task].fields["loss"] = f"{l.cpu().item():.3}"
                if progress.tasks[task].speed is not None:
                    progress.tasks[task].fields["spd"] = f"{progress.tasks[task].speed:.3}"
                progress.update(task,advance=1,description=f"[blue]Epoch:{{{E}/{epoch}}}")
            print("[!]Epoch:{},loss:{}".format(E,all_loss/len(dataloader)))
            if (E) % 10 == 0:
                torch.save(model.state_dict(),save_path+"XYLenPredictor_{}.pth".format(E))
                print("Output:",(model(nxy,rds)+1)*64)
                print("GT:",(nxylen+1)*64)

                

        