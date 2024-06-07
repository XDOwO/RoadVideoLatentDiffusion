import torch
import torch.nn as nn

# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (b, 16, 64, 64)
            nn.GroupNorm(8,16),  
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (b, 32, 32, 32)
            nn.GroupNorm(16,32),  
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (b, 64, 16, 16)
            nn.GroupNorm(32,64), 

            # nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1), # (b, 256, 16, 16)
            # nn.BatchNorm2d(256), 
            # nn.SiLU(),

            nn.Flatten(), # Flattening the output for the dense layer

            nn.Linear(64 * 16 * 16, 1024), # Intermediate reduction
            nn.GroupNorm(32,1024), 
            nn.SiLU(),
            nn.Linear(1024, 8192),   # Final output matches the specified size
            nn.GroupNorm(32,8192), 
            nn.SiLU()
        )

        # self.embedding_xys = nn.Embedding(128, 160)
        # self.embedding_xylens = nn.Embedding(128, 160)
        # self.embedding_rgbs = nn.Embedding(256, 128)
        self.embedding_xys = nn.Embedding(128, 64)
        self.embedding_xylens = nn.Embedding(128, 64)
        self.embedding_rgbs = nn.Embedding(256, 64)
        self.emb_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(112*64, 1024),
            nn.GroupNorm(32,1024), 
            nn.SiLU(),
            nn.Linear(1024, 8192),   # Final output matches the specified size
            nn.GroupNorm(32,8192), 
            nn.SiLU()
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 1024),
            nn.GroupNorm(32,1024), 
            nn.SiLU(),
            nn.Linear(1024, 8192),   # Final output matches the specified size
            nn.GroupNorm(32,8192), 
            nn.SiLU()
        )
        

    
    def forward(self, x, xys, xylens, rgbs):
        # xys_embedded = self.embedding_xys(xys.view(xys.size(0), -1))
        # xylens_embedded = self.embedding_xylens(xylens.view(xylens.size(0), -1))
        # rgbs_embedded = self.embedding_rgbs(rgbs.view(rgbs.size(0), -1))
        # concatenated = torch.cat((xys_embedded.view(xys_embedded.size(0),40,-1), xylens_embedded.view(xylens_embedded.size(0),40,-1), rgbs_embedded), dim=1)

        xys_embedded = self.embedding_xys(xys.detach().view(xys.size(0), -1))
        xylens_embedded = self.embedding_xylens(xylens.detach().view(xylens.size(0), -1))
        rgbs_embedded = self.embedding_rgbs(rgbs.detach().view(rgbs.size(0), -1))
        c = torch.cat((xys_embedded, xylens_embedded, rgbs_embedded), dim=1)
        c = self.emb_linear(c)
        x = self.encoder(x)
        x = self.linear(c+x)
        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, 16, 16)),
            nn.GroupNorm(16,32),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8,16),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(4,8),
            nn.SiLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(2,4),
            nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,3),
            nn.SiLU()
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,3),
            nn.SiLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,3),
            nn.SiLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1,3),
            nn.SiLU()
        )
        
        self.linear = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.GroupNorm(32,1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.GroupNorm(32,1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.GroupNorm(32,1024),
            nn.SiLU(),
            nn.Linear(1024, 3*128*128),
            nn.GroupNorm(32,3*128*128),
            nn.SiLU(),
            nn.Unflatten(1, (3, 128, 128)),
        )
        self.norm = nn.BatchNorm2d(3)
        self.acti = nn.Tanh()
    
    def forward(self, x):
        x1 = self.decoder(x)
        x = x1 + self.conv2d(x1) + self.linear(x)  #+ x2
        x = self.norm(x)
        x = self.acti(x)
        return x

class Encoder_Block(nn.Module):
    def __init__(self,input_c,output_c):
       super(Encoder_Block, self).__init__()
       self.norm = nn.BatchNorm2d(output_c)
       self.acti = nn.SiLU()
       self.conv = nn.Conv2d(input_c,output_c, kernel_size=3, stride=2, padding=1)  
    def forward(self,x):
       return self.acti(self.norm(self.conv(x)))     
class OldEncoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_block1 = Encoder_Block(3,16) # (b,16,64,64)
        self.encoder_block2 = Encoder_Block(16,32) # (b,3,32,32)
        self.encoder_block3 = Encoder_Block(32,64) # (b,64,16,16)
        self.encoder_block4 = Encoder_Block(64,128) # (b,128,8,8)

        self.nn = nn.Sequential(
            nn.Flatten(), # Flattening the output for the dense layer
            nn.Linear(128 * 8 * 8, 4096), # Intermediate reduction
            nn.GroupNorm(32,4096),
            nn.SiLU(),
            nn.Linear(4096, 8192),   # Final output matches the specified size
            nn.GroupNorm(32,8192), 
            nn.SiLU()
        )

        self.res = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*128*3,128*128),
            nn.GroupNorm(32,128*128), 
            nn.SiLU(),
            nn.Linear(128*128,8192),
            nn.GroupNorm(32,8192), 
            nn.SiLU()
        )

        self.norm = nn.GroupNorm(32,8192)
        self.acti = nn.SiLU()
    
    def forward(self, x):
        x0 = x
        x1 = self.encoder_block1(x)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        x4 = self.encoder_block4(x3)

        x4 = self.acti(self.norm(self.nn(x4) + self.res(x)))
        return x4,(x0,x1,x2,x3)

# Decoder
class Decoder_Block_MUL(nn.Module):
    def __init__(self,input_c,output_c):
        super(Decoder_Block_MUL, self).__init__()
        self.norm = nn.BatchNorm2d(output_c)
        self.acti = nn.SiLU()
        self.tconv = nn.ConvTranspose2d(input_c,output_c,kernel_size=3,stride=2,padding=1,output_padding=1)
        
    def forward(self,x):
        return self.acti(self.norm(self.tconv(x)))
    
class Decoder_Block_ADD(nn.Module):
    def __init__(self,input_c,output_c,kernel):
        super(Decoder_Block_ADD, self).__init__()
        self.norm = nn.BatchNorm2d(output_c)
        self.acti = nn.SiLU()
        self.tconv = nn.ConvTranspose2d(input_c,output_c,kernel_size=kernel,stride=1,padding=0, dilation=2)
        
    def forward(self,x):
        return self.acti(self.norm(self.tconv(x)))
    
class OldDecoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_mul1 = Decoder_Block_MUL(128,64)
        self.decoder_mul2 = Decoder_Block_MUL(64,32)
        self.decoder_mul3 = Decoder_Block_MUL(32,16)
        self.decoder_mul4 = Decoder_Block_MUL(16,3)

        self.decoder_add1 = Decoder_Block_ADD(128,64,5)
        self.decoder_add2 = Decoder_Block_ADD(64,32,9)
        self.decoder_add3 = Decoder_Block_ADD(32,16,17)
        self.decoder_add4 = Decoder_Block_ADD(16,3,33)

        self.last = nn.Conv2d(3,3,kernel_size=65,stride=1,padding=32) # kernel - 1 = 2 * padding

        self.conv2d = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=17, stride=1, padding=8),
            nn.GroupNorm(3,3),
            nn.SiLU(),
            nn.Conv2d(3, 3, kernel_size=33, stride=1, padding=16),
            nn.GroupNorm(3,3),
            nn.SiLU(),
            nn.Conv2d(3, 3, kernel_size=49, stride=1, padding=24),
            nn.GroupNorm(3,3),
            nn.SiLU()
        )
        
        self.linear = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.GroupNorm(32,1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.GroupNorm(32,1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.GroupNorm(32,1024),
            nn.SiLU(),
            nn.Linear(1024, 3*128*128),
            nn.GroupNorm(3,3*128*128),
            nn.SiLU(),
            nn.Unflatten(1, (3, 128, 128)),
        )
        self.last_nn = nn.Sequential(
           nn.Flatten(),
           nn.Linear(3*128*128,1024),
           nn.GroupNorm(32,1024),
           nn.SiLU(),
           nn.Linear(1024,3*128*128),
           nn.GroupNorm(32,3*128*128),
           nn.SiLU(),
           nn.Unflatten(1, (3, 128, 128)),
        )
        self.norm = nn.GroupNorm(3,3)
        self.acti = nn.SiLU()
    
    def forward(self, x, unet_sc):
        y0,y1,y2,y3 = unet_sc
        x = x.view(x.shape[0],128,8,8)
        x1 = self.decoder_mul1(x)#+y3
        x1 = self.decoder_mul2(x1)#+y2
        x1 = self.decoder_mul3(x1)#+y1
        x1 = self.decoder_mul4(x1)#+y0

        x2 = self.decoder_add1(x)#+y3
        x2 = self.decoder_add2(x2)#+y2
        x2 = self.decoder_add3(x2)#+y1
        x2 = self.decoder_add4(x2)#+y0

        x = x.view(x.shape[0],8192)
        x = x1 + self.linear(x)  + x2
        x = self.conv2d(x)
        # print(x.shape)
        return x

class CarEncoder(nn.Module):
    def __init__(self):
        super(CarEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # (b, 32, 32, 32)
            nn.GroupNorm(32,32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (b, 64, 16, 16)
            nn.GroupNorm(64,64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # (b, 128, 8, 8)
            nn.GroupNorm(128,128),
            nn.SiLU(),
            nn.Flatten(), # Flattening the output for the dense layer
            nn.Linear(128 * 8 * 8, 2048), # Intermediate reduction
            nn.SiLU(),
            nn.Linear(2048, 2048), # Intermediate reduction
            nn.SiLU()
        )
        self.emb = T_R_C_X_Embedding()
        self.last_nn = nn.Linear(448+2048,2048)
        self.norm = nn.GroupNorm(64,2048)
        self.acti = nn.Tanh()
    def forward(self, x, xy, xylen, rgb):
        # B,N,C,H,W = x.shape
        # x = x.view(B*N,C,H,W)
        # xyb = xy.view(B*N,-1)
        # xylenb = xylen.view(B*N,-1)
        # rgbb = rgb.view(B*N,-1)

        emb = self.emb(xy,xylen,rgb)
        x = self.encoder(x)
        return self.acti(self.norm(self.last_nn(torch.cat([x,emb],dim=-1))))

class CarDecoder(nn.Module):
    def __init__(self):
        super(CarDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(2048, 512 * 8 * 8), 
            nn.GroupNorm(64,512 * 8 * 8),
            nn.SiLU(),
            nn.Unflatten(1, (512, 8, 8)),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1,output_padding=1), # (b, 128, 8, 8)
            AttentionBlock2d(256,4),
            nn.GroupNorm(256,256),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1,output_padding=1), # (b, 64, 16, 16)
            nn.Conv2d(128,128,kernel_size = 3, padding = 1),
            nn.GroupNorm(128,128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1,output_padding=1),  # (b, 32, 32, 32)
            nn.GroupNorm(32,32),
            nn.SiLU(),
            nn.Conv2d(32,3,kernel_size = 3, padding = 1),
            nn.Tanh(),
        )
        
    def forward(self, x):
        x = self.decoder(x)
        return x
class AttentionBlock(nn.Module):
  def __init__(self,ch,num_heads):
    self.ch = ch
    super().__init__()
    self.norm = nn.GroupNorm(num_groups=min(ch,32), num_channels=ch)
    self.qkv = nn.Conv2d(ch,ch*3,1)
    self.attention = nn.MultiheadAttention(embed_dim=ch,num_heads=num_heads,dropout=0.2,batch_first=True)
    self.out = nn.Conv2d(ch,ch,kernel_size=3,padding=1,stride=1)
  def forward(self,x,*unused):
    x = self.norm(x)
    qkv = self.qkv(x)
    qkv = qkv.permute(0,2,1)
    q,k,v = torch.split(qkv,self.ch,2)
    x,_ = self.attention(q,k,v)
    x = x.permute(0,2,1)
    return self.out(x)
class AttentionBlock2d(nn.Module):
  def __init__(self,ch,num_heads):
    self.ch = ch
    super().__init__()
    self.norm = nn.GroupNorm(num_groups=min(ch,32), num_channels=ch)
    self.qkv = nn.Conv1d(ch,ch*3,1)
    self.attention = nn.MultiheadAttention(embed_dim=ch,num_heads=num_heads,dropout=0.2,batch_first=True)
    self.out = nn.Conv2d(ch,ch,1)
  def forward(self,x,*unused):
    B,C,H,W = x.shape
    x = x.reshape(B,C,-1)
    x = self.norm(x)
    qkv = self.qkv(x)
    qkv = qkv.permute(0,2,1)
    q,k,v = torch.split(qkv,self.ch,2)
    x,_ = self.attention(q,k,v)
    x = x.permute(0,2,1)
    x = x.view(B,C,H,W)
    return self.out(x)
class T_R_C_X_Embedding(nn.Module):
  def __init__(self, xy_nums=8, xy_max_size=256, input_dim=1024,output_dim=512):
    assert output_dim >= xy_nums, "output_dim({}) >= xy_nums({}) not achieved".format(output_dim,xy_nums)
    assert output_dim % (xy_nums)==0, "output_dim({}) have to be divisible by xy_nums({})".format(output_dim,xy_nums)
    super(T_R_C_X_Embedding, self).__init__()
    self.embedding = nn.Embedding(xy_max_size, output_dim//xy_nums)
    self.output_dim = output_dim
  def forward(self, xys,xylens,rgbs, max_period=10000):
    '''
    Create embedding from timestep,empty road,car and cars' coordination(xy)

    :param timesteps: 1-D tensor. The shape should be the same as batch size.
    :param rds: 2-D tensor. The shape should be (batch,1024)
    :param cars: 2-D tensor. The shape should be (batch,1024)
    :param xys: 2/3-D tensor. The shape should be (batch,32) or (batch,16,2)
    '''
    
    if len(xys.shape) == 3:
      B,N,C = xys.shape
      xys = xys.view(B,N*C)
    if len(rgbs.shape) == 3:
      B,N,C = rgbs.shape
      rgbs = rgbs.view(B,N*C)
    if len(xylens.shape) == 3:
      B,N,C = xylens.shape
      xylens = xylens.view(B,N*C)
    everything = torch.cat((xys,xylens,rgbs),dim=-1)
    everything = self.embedding(everything).view(xys.shape[0], -1)
    return everything

class Mixer(nn.Module):
    def __init__(self):
        super(Mixer, self).__init__()
        # self.attention_road = AttentionBlock(8192,1)
        # self.attention_car = AttentionBlock(2048,1)
        self.emb = T_R_C_X_Embedding()
        self.nn1 = nn.Linear(8192+1024,2048)
        self.nnc1 = nn.Linear(8192,1024)
        self.nnc2 = nn.Linear(1024,1024)
        self.nn2 = nn.Linear(2048,1024)
        self.nn2_2 = nn.Linear(1024,1024)
        self.nn3 = nn.Linear(1024,2048)
        self.nn4 = nn.Linear(2048,8192)
        self.norm = nn.GroupNorm(32,8192)
        self.norm2 = nn.GroupNorm(32,1024)
        self.acti = nn.SiLU()
        self.nn = nn.Sequential(
            self.nn1,
            nn.GroupNorm(32,2048),
            self.acti,
            self.nn2,
            nn.GroupNorm(32,1024),
            self.acti,
            self.nn2_2,
            nn.GroupNorm(32,1024),
            self.acti,
            self.nn2_2,
            nn.GroupNorm(32,1024),
            self.acti,
            self.nn2_2,
            nn.GroupNorm(32,1024),
            self.acti,
            self.nn2_2,
            nn.GroupNorm(32,1024),
            self.acti,
            self.nn3,
            nn.GroupNorm(32,2048),
            self.acti,
            self.nn4,
            nn.GroupNorm(32,8192),
            self.acti
        )
        self.nn_car = nn.Sequential(
            self.nnc1,
            nn.GroupNorm(32,1024),
            self.acti,
            self.nnc2,
            nn.GroupNorm(32,1024),
            self.acti,
            self.nnc2,
            nn.GroupNorm(32,1024),
            self.acti,
            self.nnc2,
            nn.GroupNorm(32,1024),
            self.acti,
            self.nnc2,
            nn.GroupNorm(32,1024),
            self.acti,
        )
    def forward(self,car1d,emp1d,xy,xylen,rgb):
        emb = self.emb(xy,xylen,rgb)
        emb_car = torch.cat((emb,car1d),dim=-1)
        emb_car = self.acti(self.norm2(self.nn_car(emb_car) + car1d))
        # emb_car = emb_car.unsqueeze(-1)
        # emp1d = emp1d.unsqueeze(-1)
        # emb_car = self.attention_car(emb_car)
        # road = self.attention_road(emp1d)
        # emb_car = emb_car.squeeze(-1)
        # road = road.squeeze(-1)
        xx = self.nn(torch.cat([emb_car,emp1d],dim=-1))
        # x = torch.cat([emb_car,torch.zeros((emb_car.shape[0],8192-1024),device=emb_car.device)],dim=-1)
        return self.acti(self.norm(xx))

