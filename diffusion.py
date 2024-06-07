#diffusion.py
import torch
import math
import os
import torch.nn as nn
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from torchvision import transforms
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn, MofNCompleteColumn,TextColumn,track
from random import randint
import torch.nn.functional as F
import copy

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.original = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.original[name]
class Diffusion:
  def __init__(self,T,model,encoder,decoder,c_encoder,c_decoder,save_path,schedule_type="linear",device="cuda:0",scale=None):
    self.device = device
    # self.device = "cpu"
    self.T = T
    if schedule_type == "linear":
      self.betas = self.linear_schedule(T).to(self.device)
    elif schedule_type == "cosine":
      self.betas = self.cosine_schedule(T).to(self.device)
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas,dim = 0).to(self.device)
    self.alphas_cumprod_prev = torch.cat((torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1])).to(self.device)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)
    self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
    self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
    self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
    self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)).to(self.device)
    self.posterior_log_variance_clipped = torch.log(torch.cat((self.posterior_variance[1:2], self.posterior_variance[1:]))).to(self.device)
    self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).to(self.device)
    self.posterior_mean_coef2 = (
        (1.0 - self.alphas_cumprod_prev)
        * torch.sqrt(self.alphas)
        / (1.0 - self.alphas_cumprod)
    ).to(self.device)
    self.save_path = save_path
    self.model = model.to(self.device)
    self.encoder = encoder.to(self.device)
    self.decoder = decoder.to(self.device)
    self.c_encoder = c_encoder.to(self.device)
    self.c_decoder = c_decoder.to(self.device)
    self.CarTensor2Img = transforms.Compose([
        transforms.Lambda(lambda t: self.c_decoder(t).view(3,64,64)),
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.ToPILImage(),
    ])
  def linear_schedule(self,T,scale=None,start=0.0001,end=0.02):
    '''The schedule is from improved diffusion'''
    if scale == None:
      scale = 1000/T
    return torch.linspace(start*scale,end*scale,T,dtype=torch.float32)
  def cosine_schedule(self,T,s=0.008,max_beta=0.9999):
    '''The schedule is from improved diffusion'''
    def alpha_bar(t):
      return math.cos((t+0.008) / 1.008 * math.pi /2) ** 2
    betas = []
    for i in range(T):
      t1 = i / T
      t2 = (i+1) / T
      betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas,dtype=torch.float32)

  def forward_process(self, x_0, t, noise=None):
    '''
    Given x_0,t , get x_t
    '''
    if noise == None:
      noise = torch.randn_like(x_0).to(self.device)
    assert noise.shape == x_0.shape, "Noise shape{} should be the same as x shape{}".format(noise.shape,x_0.shape)
    # print(t.shape,x_0.shape,noise.shape,self.sqrt_alphas_cumprod.shape,self.sqrt_one_minus_alphas_cumprod.shape,(x_0*self.sqrt_alphas_cumprod[t]).shape,(noise*self.sqrt_one_minus_alphas_cumprod[t]).shape)
    return x_0*extract(self.sqrt_alphas_cumprod,t,x_0.shape) + noise*extract(self.sqrt_one_minus_alphas_cumprod,t,x_0.shape),noise

  def show_forward_process_img(self,x,noise = None):
    plt.figure(figsize=(1024/72, 576/144))
    plt.subplot(1,11,1)
    t = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.ToPILImage(),
    ])
    plt.imshow(t(x))
    if noise == None:
      noise = torch.randn_like(x)
    for i in range(2,12):
      plt.subplot(1,11,i)
      noised_x,noise = self.forward_process(x,t = torch.full((x.shape[0],),self.T/10*(i-1)-1,dtype=torch.long).to(self.device),noise=noise)
      plt.imshow(t(noised_x))
      plt.axis("off")
    
    plt.savefig(os.path.join(self.save_path,"plt","forward.png"))
    plt.close()

    return noised_x,noise
  
  @torch.no_grad
  def backward_step(self,x_t,t,rds = None,cars = None,xys = None,xylens = None,rgbs = None,indices = None,eps = None):
    '''
    Given x_t,t,and else, get x_(t-1)
    '''
    
    noise = torch.randn_like(x_t)
    eps = eps if eps is not None else self.model(x_t,t,rds,cars,xys,xylens,rgbs,indices)
    pred_xstart = extract(self.sqrt_recip_alphas,t,x_t.shape)*x_t - extract(self.sqrt_recipm1_alphas_cumprod,t,x_t.shape)*eps
    mean = extract(self.posterior_mean_coef1, t, x_t.shape) * pred_xstart + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
    log_variance = extract(torch.log(torch.cat((self.posterior_variance[1:2], self.betas[1:]),dim=0)), t, x_t.shape)    
    nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
    return mean + nonzero_mask * torch.exp(0.5*log_variance)*noise
    # variance = self.posterior_variance[t]
    # betas_t = self.betas[t]
    # sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
    # sqrt_recip_alphas_t =self.sqrt_recip_alphas[t]
    # mean = self.model(x_t,t,rds,cars,xys) if n2 == None else n2
    # mean = sqrt_recip_alphas_t*(x_t-betas_t * mean / sqrt_one_minus_alphas_cumprod_t)
    # if t==0:
    #   return mean
    # else:
    #   return mean+torch.sqrt(variance)*noise
  
  @torch.no_grad
  def backward_process(self,shape,rds,cars,xys,xylens,rgbs,indices,noise=None,save_intermidate = False):
    if noise == None:
      noise = torch.randn(*shape).to(self.device)
      sample = noise
    else:
      sample = noise
    intermidate = []
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
        transient=True  # 任務完成後自動隱藏進度條
    ) as progress:
      task = progress.add_task(f"Backward processing",total=self.T,loss=0,spd=0)
      for i in range(0,self.T)[::-1]:
        if save_intermidate:
          intermidate.append(sample[0] if len(sample.shape)==4 else sample)
        t = torch.tensor([i]*shape[0]).to(self.device)
        sample = self.backward_step(sample,t,rds,cars,xys,xylens,rgbs,indices)
        if progress.tasks[task].speed is not None:
          progress.tasks[task].fields["spd"] = f"{progress.tasks[task].speed:.3}"
        progress.update(task,advance=1)
    intermidate.append(sample[0] if len(sample.shape)==4 else sample)
    return sample if not save_intermidate else intermidate

  @torch.no_grad
  def show_backward_process_img(self,shape,rds,cars,xys,xylens,rgbs,noise=None):
    lis = self.backward_process(shape,rds,cars,xys,xylens,rgbs,noise,True)
    plt.figure(figsize=(1024/72, 576/144))
    
    # t2 =transforms.Compose([
    #     transforms.Lambda(lambda t: self.c_decoder(t)),
    #     transforms.Lambda(lambda t: (t + 1) / 2),
    #     transforms.ToPILImage(),
    # ])
    plt.subplot(1,11,11)
    plt.imshow(self.CarTensor2Img(lis[0]))
    plt.axis("off")
    for i in range(1,11):
      plt.subplot(1,11,i)
      plt.imshow(self.CarTensor2Img(lis[int(self.T-self.T/10*(i-1)-1)]))
      plt.axis("off")

    plt.savefig(os.path.join(self.save_path,"plt","backward.png"))
    self.CarTensor2Img(lis[-1]).save(os.path.join(self.save_path,"test_res","res.png"))
    plt.close()
    return lis

  def weighted_mse(self,targets,predicts,xys,xylens,full_weight=1000,mode="road"):
    assert targets.shape[0]==xys.shape[0], "target and xy batch should be the same."
    mask = torch.full_like(targets,1) if mode == "road" else torch.full_like(targets, -1)
    w,h = targets.shape[2:]
    for b in range(targets.shape[0]):
      for xy_n in range(xys.shape[1]):
        x,y = xys[b][xy_n]
        x_rad, y_rad = xylens[b][xy_n]
        x_rad //= 2
        y_rad //= 2

        if x==y==0:
          continue
        mask[b][:,max(0,y-y_rad):min(y+y_rad,h):,max(x-x_rad,0):min(x+x_rad,w)] = -1 if mode == "road" else 1
    
    # t2 =transforms.Compose([
    #     transforms.Lambda(lambda t: (t + 1) / 2),
    #     transforms.ToPILImage(),
    # ])
    targets[mask == -1] = -1
    predicts[mask == -1] = -1
    # t2(targets[0]).save("/home/fish/OursDiffusion/testimg/target.png")
    # t2(mask[0]).save("/home/fish/OursDiffusion/testimg/mask.png")
    # assert 1==0
        
    # xt,_ = self.forward_process(gt,ts,targets)
    # xt_minus_1,_ = self.forward_process(gt,torch.clamp(ts,min=0),noise=targets)
    # predicts = self.backward_step(xt,ts,eps = predicts)
    return torch.mean(((targets-predicts)**2)*full_weight)
  def train_model(self,epoch,dataloader,test_dataloader,lr,ratio=1,save_interval = 20):
    def cut_image(im,xy,xylen):
      

      B, C, H, W = im.shape
      patches = []
      _, N , _  = xy.shape
      for i in range(B):
          for j in range(N):
            x_center, y_center = xy[i][j]
            x_length, y_length = xylen[i][j]//2
            if(x_length==0 or y_length==0) :
              continue
            x_start = max(x_center - x_length, 0) 
            x_end = min(x_center + x_length, W) 
            y_start = max(y_center - y_length, 0) 
            y_end = min(y_center + y_length, H) 

            patch = im[i, :, y_start:y_end, x_start:x_end].unsqueeze(0)
            resized = nn.functional.interpolate(patch,(64,64),mode='bilinear')
            patches.append(resized)
            
      
      patches = torch.stack(patches).view(-1,3,64,64)      
      return nn.functional.interpolate(patches,(64,64))
    import torch.optim as optim
    import os.path
    import torch.optim.lr_scheduler as lr_scheduler
    if not os.path.exists(self.save_path):
      os.makedirs(os.path.join(self.save_path,"plt"),exist_ok=True)
      os.makedirs(os.path.join(self.save_path,"model"),exist_ok=True)
      os.makedirs(os.path.join(self.save_path,"test_res"),exist_ok=True)

    loss = nn.MSELoss()
    opt = optim.Adam(self.model.parameters(),lr = lr)
    # scheduler = lr_scheduler.MultiStepLR(opt, milestones=[10,20,30,40], gamma=0.5)
    noise = None
    
    for E in range(epoch):
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
        i = 0
        all_loss_r = 0
        all_loss_c = 0
        all_loss = 0
        ema = EMA(self.model,decay=0.999)
        self.model.train()
        task = progress.add_task(f"[blue]Epoch:{{{E}/{epoch}}}",total=len(dataloader),loss=0,spd=0)
        for gt,car,emp,xy,xylen,rgbs in dataloader:
          # t = torch.randint(0,self.T,size=(gt.shape[0],),dtype=torch.long)
          i+=1
          gt, emp, xy, car, xylen, rgbs = gt.to(self.device), emp.to(self.device), xy.to(self.device), car.to(self.device), xylen.to(self.device) , rgbs.to(self.device)
          emp_e = self.encoder(emp,xy,xylen,rgbs)
          
          gtcar = cut_image(gt,xy,xylen)
          B,N,C,H,W = car.shape
          ncar = car.view(B*N,C,H,W)
          B,N,C = xy.shape
          nxy = xy.view(B*N,C)
          B,N,C = xylen.shape
          nxylen = xylen.view(B*N,C)
          B,N,C = rgbs.shape
          nrgbs = rgbs.view(B*N,C)
          m0 = nxylen[:,0] > 1
          m1 = nxylen[:, 1] > 1

          combined_mask = torch.logical_and(m0, m1)
          indices = torch.nonzero(combined_mask,as_tuple=True)
          nxy = nxy[indices]
          nxylen = nxylen[indices]
          fake_nxylen = torch.zeros_like(nxylen)
          nrgbs = nrgbs[indices]
          ncar = ncar[indices]
          encoded_car = self.c_encoder(ncar,nxy,nxylen,nrgbs)
          encoded_gtcar = self.c_encoder(gtcar,nxy,nxylen,nrgbs)
          cpy_encoded_car = encoded_car.clone().detach()
          nrgbs = nrgbs
          # cpy_encoded_car[:] = 0 #Maybe too much infos is not a good idea
          t = torch.randint(low=0, high=self.T, size=(encoded_car.shape[0],), dtype=torch.long).to(self.device)
          noise = torch.randn_like(encoded_gtcar).to(self.device)
          x_t,_ = self.forward_process(encoded_gtcar,t,noise)
          # Calculate car loss first
          output = self.model(x_t,t,emp_e,cpy_encoded_car,nxy,fake_nxylen,nrgbs,indices)
          l = loss(output,noise)
          l.backward()
          all_loss+=l.cpu().item()
          # loss_c = self.weighted_mse(output,noise,xy,xylen,1,"car")
          # loss_c_cpu = loss_c.cpu().item()
          # loss_c.backward()
          # opt.step()
          # all_loss_c+=loss_c_cpu
          # opt.zero_grad()

          # Calculate road loss second
          # output = self.model(x_t,t,emp,car,xy)
          # loss_r = self.weighted_mse(output,noise,xy,xylen,ratio,"road")
          # loss_r_cpu = loss_r.cpu().item()
          # loss_sum = loss_r+loss_c
          # loss_sum.backward()
          # loss_r.backward()
          opt.step()
          opt.zero_grad()
          # all_loss_r+=loss_r_cpu
          # loop.set_description(f'Epoch [{E}/{epoch}]')
          # loop.set_postfix(loss=loss_r_cpu+loss_c_cpu)
          # loop.set_postfix(loss=l.cpu().item())
          ema.update()
          progress.tasks[task].fields["loss"] = f"{l.cpu().item():.3}"
          if progress.tasks[task].speed is not None:
            progress.tasks[task].fields["spd"] = f"{progress.tasks[task].speed:.3}"
          progress.update(task,advance=1,description=f"[blue]Epoch:{{{E}/{epoch}}}")

      # print("Avg loss_road at epoch {}:".format(E),all_loss_r/len(dataloader))
      # print("Avg loss_car at epoch {}:".format(E),all_loss_c/len(dataloader))
      print("Train: Avg loss at epoch {}:".format(E),all_loss/len(dataloader))
      ### test
      all_loss = 0
      self.model.eval()
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
        i = 0
        all_loss_r = 0
        all_loss_c = 0
        all_loss = 0
        ema = EMA(self.model,decay=0.999)
        self.model.train()
        task = progress.add_task(f"[blue]Epoch:{{{E}/{epoch}}}",total=len(test_dataloader),loss=0,spd=0)
        for gt,car,emp,xy,xylen,rgbs in test_dataloader:
          gt, emp, xy, car, xylen, rgbs = gt.to(self.device), emp.to(self.device), xy.to(self.device), car.to(self.device), xylen.to(self.device) , rgbs.to(self.device)
          emp_e = self.encoder(emp,xy,xylen,rgbs)
          
          gtcar = cut_image(gt,xy,xylen)
          B,N,C,H,W = car.shape
          ncar = car.view(B*N,C,H,W)
          B,N,C = xy.shape
          nxy = xy.view(B*N,C)
          B,N,C = xylen.shape
          nxylen = xylen.view(B*N,C)
          B,N,C = rgbs.shape
          nrgbs = rgbs.view(B*N,C)
          m0 = nxylen[:,0] > 1
          m1 = nxylen[:, 1] > 1

          combined_mask = torch.logical_and(m0, m1)
          indices = torch.nonzero(combined_mask,as_tuple=True)
          nxy = nxy[indices]
          nxylen = nxylen[indices]
          fake_nxylen = torch.zeros_like(nxylen)
          nrgbs = nrgbs[indices]
          ncar = ncar[indices]
          encoded_car = self.c_encoder(ncar,nxy,nxylen,nrgbs)
          encoded_gtcar = self.c_encoder(gtcar,nxy,nxylen,nrgbs)
          cpy_encoded_car = encoded_car.clone().detach()
          nrgbs = nrgbs
          nxy = nxy
          # cpy_encoded_car[:] = 0 #Maybe too much infos is not a good idea
          t = torch.randint(low=0, high=self.T, size=(encoded_car.shape[0],), dtype=torch.long).to(self.device)
          noise = torch.randn_like(encoded_gtcar).to(self.device)
          x_t,_ = self.forward_process(encoded_gtcar,t,noise)
          # Calculate car loss first
          output = self.model(x_t,t,emp_e,cpy_encoded_car,nxy,fake_nxylen,nrgbs,indices)
          l = loss(output,noise)
          l.backward()
          all_loss+=l.cpu().item()
          progress.tasks[task].fields["loss"] = f"{l.cpu().item():.3}"
          if progress.tasks[task].speed is not None:
            progress.tasks[task].fields["spd"] = f"{progress.tasks[task].speed:.3}"
          progress.update(task,advance=1,description=f"[blue]Epoch:{{{E}/{epoch}}}")
      print("Test: Avg loss at epoch {}:".format(E),all_loss/len(test_dataloader))
      
      ### generate image
      ema.apply_shadow()
      emp=emp[0].unsqueeze(0)
      car=car[0].unsqueeze(0)
      xy=xy[0].unsqueeze(0)
      gt=gt[0].unsqueeze(0)
      xylen=xylen[0].unsqueeze(0)
      rgbs=rgbs[0].unsqueeze(0)
      emp_e = self.encoder(emp,xy,xylen,rgbs)
      B,N,C,H,W = car.shape
      ncar = car.view(B*N,C,H,W)
      B,N,C = xy.shape
      nxy = xy.view(B*N,C)
      B,N,C = xylen.shape
      nxylen = xylen.view(B*N,C)
      B,N,C = rgbs.shape
      nrgbs = rgbs.view(B*N,C)
      m0 = nxylen[:,0] > 1
      m1 = nxylen[:, 1] > 1

      combined_mask = torch.logical_and(m0, m1)
      indices = torch.nonzero(combined_mask,as_tuple=True)
      nxy = nxy[indices]
      #clear nxy
      # nxy = torch.zeros_like(nxy)
      nxylen = nxylen[indices]
      fake_nxylen = torch.zeros_like(nxylen)
      nrgbs = nrgbs[indices]
      ncar = ncar[indices]
      encoded_car = self.c_encoder(ncar,nxy,nxylen,nrgbs)
      cpy_encoded_car = encoded_car.clone().detach()
      nxy = nxy
      nrgbs = nrgbs
      sample = self.backward_process(encoded_car.shape,emp_e,encoded_car,nxy,fake_nxylen,nrgbs,indices)
      dcar = self.c_decoder(sample).view(1,*ncar.shape)
      add_car(emp[0],dcar[0],xy[0],xylen[0])
      t = transforms.Compose([
              transforms.Lambda(lambda t: (t + 1) / 2),
              transforms.ToPILImage(),
          ])
      im = torch.cat([gt[0],emp[0]],dim=-1)
      t(im).save("./test_result/gt_{}.png".format(E))
      print("Saved!")
      ema.restore()
      if E % save_interval == 0:
        ema.apply_shadow()
        torch.save(self.model.state_dict(),os.path.join(self.save_path,"model","OurDiffusion_2dataset_{}.pth".format(E//save_interval)))
        print("Model saved!")
      # scheduler.step()
        ema.restore()
    
def extract(a, t, x_shape):
        """
            from lucidrains' implementation
                https://github.com/lucidrains/denoising-diffusion-pytorch/blob/beb2f2d8dd9b4f2bd5be4719f37082fe061ee450/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L376
        """
        b, *_ = t.shape
        # print(t)
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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
