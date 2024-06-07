#dataset.py
import torch
from PIL import Image
from torchvision import datasets, transforms
import os
import csv
from torch.utils.data import DataLoader
class DiffusionDataset(torch.utils.data.Dataset):
  def __init__(self,root_dir,rd_transform=None,car_transform=None,max_car=16):
    self.root_dir = root_dir
    self.gt_dir_list = sorted([os.path.join(root_dir,"gt",i) for i in os.listdir(os.path.join(root_dir,"gt"))])
    self.emp_dir_list = sorted([os.path.join(root_dir,"emp_rd",i) for i in os.listdir(os.path.join(root_dir,"emp_rd"))])
    self.cars_csv_dir = os.path.join(root_dir,"cars","cars.csv")
    self.cars_dir_list = sorted([os.path.join(root_dir,"cars",i) for i in os.listdir(os.path.join(root_dir,"cars"))])
    self.max_car = max_car
    self.emp_rd_len = None
    self.cars_csv_len = None
    self.emp_rd_data_list = None
    self.cars_data_list = None


    if rd_transform is None:
      self.rd_transform = transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    else:
      self.rd_transform = rd_transform

    if car_transform is None:
      self.car_transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    else:
      self.car_transform = car_transform

  def load_img(self,img_path):
    img = Image.open(img_path)
    return img

  def __len__(self):
    # if self.emp_rd_len is None:
      # with open(self.emp_rd_csv_dir,'r') as csv_file:
      #   csv_reader = csv.reader(csv_file)
      #   self.emp_rd_len = sum(1 for row in csv_reader)
    gt_len = len(self.gt_dir_list)
    # cars_len = len(self.cars_dir_list)
    # assert gt_len == self.emp_rd_len == cars_len == self.cars_csv_len, "All sub-directory must contain the same number of files or directories while gt_len = {}, emp_rd_len={}, cars_csv_len={}, and cars_len={}".format(gt_len,self.emp_rd_len,self.cars_csv_len,cars_len)
    return gt_len

  def __getitem__(self,idx):

    gt_img_path = self.gt_dir_list[idx]
    gt_img = self.load_img(gt_img_path)
    gt_img = self.rd_transform(gt_img)
    emp_rd_data_path = self.emp_dir_list[idx]
    emp_rd_data = self.load_img(emp_rd_data_path)
    emp_rd_data = self.rd_transform(emp_rd_data)
    
    # if self.emp_rd_data_list is None:
    #   with open(self.emp_rd_csv_dir,'r') as csv_file:
    #     csv_reader = csv.reader(csv_file)
    #     self.emp_rd_data_list = [(lambda x:[float(i) for i in x])(i) for i in csv_reader]
    # emp_rd_data=torch.tensor(self.emp_rd_data_list[idx])
    
    car_imgs = torch.zeros((self.max_car,3,64,64))
    xys = torch.zeros((self.max_car,2),dtype=torch.int)
    xylens = torch.zeros((self.max_car,2),dtype=torch.int)
    rgbs = torch.zeros((self.max_car,3),dtype=torch.int)
    # try:
    csvf = [file for file in os.listdir(self.cars_dir_list[idx]) if file[-4:]==".csv"][0]
    cars_csv_path = os.path.join(self.cars_dir_list[idx],csvf)
    # except:
    #   cars_csv_path = os.path.join(self.cars_dir_list[idx],"xy.csv")
    with open(cars_csv_path,'r') as csv_file:
      csv_reader = csv.reader(csv_file)
      for i,item in enumerate(csv_reader):
        car_img_path,x,y,x_length,y_length,r,g,b = item
        xy_tensor = torch.tensor((int(x)*127//639,int(y)*127//639),dtype=torch.int)
        xylen_tensor = torch.tensor((int(x_length)*127//639,int(y_length)*127//639),dtype=torch.int)
        rgb_tensor = torch.tensor((int(r),int(g),int(b)),dtype=torch.int)
        car_img_path = os.path.join(self.cars_dir_list[idx],car_img_path)
        car_img = self.load_img(car_img_path)
        car_img = self.car_transform(car_img)

        xys[i] = xy_tensor
        xylens[i] = xylen_tensor
        rgbs[i] = rgb_tensor
        car_imgs[i] = car_img
      # print(gt_img.shape)
      # print(emp_rd_data.shape)
      # print(xys.shape)
      # print(cars_data.shape)
    return gt_img,car_imgs,emp_rd_data,xys,xylens,rgbs



if __name__ == "__main__":
  dataset = DiffusionDataset("/dataset")
  dataloader=DataLoader(dataset,batch_size=64,shuffle = True)
  for gt,emp,xy,car in dataloader:
    print(gt.shape,emp.shape,car.shape,xy.shape)
    print(car[1][0],xy[1][0])

