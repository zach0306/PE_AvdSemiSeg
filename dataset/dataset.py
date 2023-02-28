import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from torchvision import transforms
from utils.transforms import OneHotEncode
import pickle
import pydicom
import torchvision
from torch.autograd import Variable
import cv2


transform = transforms.Compose([
    #transforms.Resize((256, 192)),
    transforms.ToTensor() #PIL Image/ndarray (H,W,C) [0,255] to tensor (C,H,W) [0.0,1.0]
    ]) 

def load_image(file):
    return Image.open(file)

def read_img_list(filename):
    with open(filename) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return np.array(img_list)

def crop_center(img,cropx,cropy):
    x,y = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2    
    return img[startx:startx+cropx, starty:starty+cropy]

def rescale_pixelarray(dataset):
    image = dataset.pixel_array
    rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept
    return rescaled_image

def default_loader(path):
  
    image = pydicom.dcmread(path)
    image = rescale_pixelarray(image)
    image = crop_center(image, 400, 400)
    #image_o = image.copy()
    #print(np.min(image))
    image[image>240] = 240
    image[image<-160] = -160
    #print(np.min(image))
    #image[image>1624] = 1624
    #image = (image/2047.5)-1
    #image = (2*image/1624)-1
    #image = (image/812)-1
    #print(np.max(image))
    image_norm = 2*(image - np.min(image))/(np.max(image)-np.min(image))-1

    img_tensor = torch.tensor(image_norm,dtype=torch.float64)
    try:
        img_tensor=img_tensor.permute(2,0,1)

    except:
        img_tensor=img_tensor.unsqueeze(0)

    #img_tensor=img_tensor.unsqueeze(0)
    #print(img_tensor.shape)
    return img_tensor

def default_loader_mask(path):
    image = Image.open(path).convert('L')
    image=np.array(image)
    image = crop_center(image, 400, 400)
    #print(np.max(image))

    #image=image//255.
    #img_tensor=np.array(image).astype(np.uint8)
    img_tensor = transform(image)
    #img_tensor = torch.from_numpy(image)
    #img_tensor =  Variable(torch.from_numpy(image)).type(torch.long)
    #print(img_tensor.shape)
    #cv2.imwrite('a.jpg', image.astype('uint8'))*

    img_tensor=img_tensor.squeeze(0)
    #print(img_tensor.shape)
    return img_tensor

class train_labeled(Dataset):
    def __init__(self, image_path, mask_path, transforms = None):
        self.images = image_path
        self.masks = mask_path
        self.transforms = transforms
    
    def Combine_mask(self, gt0):
        mask = np.full(gt0.shape[0:2], 0, dtype = int)
        #print('shape',mask.shape, gt0.shape, gt1.shape, gt2.shape)
        #mask[gt2 == 1] = 2
        mask[gt0 == 1] = 1
        
        return mask

    def __getitem__(self, index):
        img = self.images[index]
        label = self.masks[index]
        img_path = img
        
        img = default_loader(img)    
        label = default_loader_mask(label)
        #print(img.max(), img.min())
        #print(GT.max(), GT.min())
        #if self.transforms != None :
        #    img, label = self.transforms(img, label)

        #ohlabel = OneHotEncode()(label)
        ohlabel = label.unsqueeze(0)
        #print('ohlabel', ohlabel.shape)
        #print('label', label.shape)

        return img.float(), label, ohlabel, img_path

    def __len__(self):
        return len(self.images)


class train_unlabeled(Dataset):
    def __init__(self, image_path, transforms = None):
        self.images = image_path
        self.transforms = transforms
    
    def Combine_mask(self, gt0):
        mask = np.full(gt0.shape[0:2], 0, dtype = int)
        #print('shape',mask.shape, gt0.shape, gt1.shape, gt2.shape)
        #mask[gt2 == 1] = 2
        mask[gt0 == 1] = 1
        
        return mask

    def __getitem__(self, index):
        img = self.images[index]
        img_path = img
        img = default_loader(img)
        
        #if self.transforms != None :
        #    img = self.transforms(img)
              
        return img.float(), img_path

    def __len__(self):
        return len(self.images)


'''
class PascalVOC(Dataset):

    TRAIN_LIST = "lists/train.txt"
    VAL_LIST = "lists/val.txt"

    def __init__(self, root, data_root, img_transform = Compose([]), label_transform=Compose([]), co_transform=Compose([]), train_phase=True,split=1,labeled=True,seed=0):
        np.random.seed(100)
        self.n_class = 1
        self.root = root
        self.data_root = data_root
        self.images_root = os.path.join(self.data_root, 'img')
        self.labels_root = os.path.join(self.data_root, 'cls')
        self.img_list = read_img_list(os.path.join(self.root,'datasets',self.TRAIN_LIST)) \
                        if train_phase else read_img_list(os.path.join(self.root,'datasets',self.VAL_LIST))
        self.split = split
        self.labeled = labeled
        n_images = len(self.img_list)
        self.img_l = np.random.choice(range(n_images),int(n_images*split),replace=False) # Labeled Images
        self.img_u = np.array([idx for idx in range(n_images) if idx not in self.img_l],dtype=int) # Unlabeled Images
        if self.labeled:
            self.img_list = self.img_list[self.img_l]
        else:
            self.img_list = self.img_list[self.img_u]
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.co_transform = co_transform
        self.train_phase = train_phase

    def __getitem__(self, index):
        filename = self.img_list[index]

        with open(os.path.join(self.images_root,filename+'.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(os.path.join(self.labels_root,filename+'.png'), 'rb') as f:
            label = load_image(f).convert('P')

        image, label = self.co_transform((image,label))
        image = self.img_transform(image)
        label = self.label_transform(label)
        ohlabel = OneHotEncode()(label)

        return image, label, ohlabel

    def __len__(self):
        return len(self.img_list)
'''



