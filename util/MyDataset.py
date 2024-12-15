import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import random
from torchvision import transforms
from torch.utils import data
from PIL import Image


class MyDataset(data.Dataset):
    def __init__(self, data_dir, datainfo_path, transform, patch_num, crop_size = 224, img_length_read = 6, is_train = True):
        super(MyDataset, self).__init__()
        dataInfo = pd.read_csv(datainfo_path, header = 0, sep=',', index_col=False, encoding="utf-8-sig")
        name_model_list = dataInfo['model'].tolist()
        name_prompt_list = dataInfo['prompt'].tolist()
        mos_alignment_list = dataInfo['Alignment'].tolist()
        mos_geometry_list = dataInfo['Geometry'].tolist()
        mos_texture_list = dataInfo['Texture'].tolist()
        mos_overall_list = dataInfo['Overall'].tolist()
        
        
        self.crop_size = crop_size
        self.patch_num = patch_num
        self.data_dir = data_dir
        self.transform = transform
        self.img_length_read = img_length_read
       
        
        self.is_train = is_train
        
        self.name_model = [item for item in name_model_list for _ in range(patch_num)]
        self.name_prompt = [item for item in name_prompt_list for _ in range(patch_num)]
        self.mos_alignment = [item for item in mos_alignment_list for _ in range(patch_num)]
        self.mos_geometry = [item for item in mos_geometry_list for _ in range(patch_num)]
        self.mos_texture = [item for item in mos_texture_list for _ in range(patch_num)]                
        self.mos_overall = [item for item in mos_overall_list for _ in range(patch_num)]

        self.length = len(self.name_model)

        
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        name_model = self.name_model[idx]
        name_prompt = self.name_prompt[idx]
        name_prompt = name_prompt.replace(" ", "_")
        data_dir = self.data_dir
    
        img_channel = 3
        img_height_crop = self.crop_size
        img_width_crop = self.crop_size
        img_length_read = self.img_length_read       
        
        img_transformed = torch.zeros([img_length_read, img_channel, img_height_crop, img_width_crop])
        img_read_index = 0
        
        for i in range(img_length_read):
      
            imge_name = os.path.join(data_dir, name_model, name_prompt, 'rendered_view_' + str(i) + '.png')
            if os.path.exists(imge_name):
                img = Image.open(imge_name)
                img = img.convert('RGB')
                img = transforms.ToTensor()(img)
                img = self.transform(img)
                img_transformed[i] = img
                img_read_index += 1
            else:
                print(imge_name)
                print('Image do not exist!')

        if img_read_index < img_length_read:
            for j in range(img_read_index, img_length_read):
                img_transformed[j] = img_transformed[img_read_index-1]

        mos_alignment = self.mos_alignment[idx]
        mos_geometry = self.mos_geometry[idx]
        mos_texture = self.mos_texture[idx]
        mos_overall = self.mos_overall[idx]
        
        mos = torch.FloatTensor(np.array([mos_alignment,mos_geometry,mos_texture,mos_overall]))
        
        prompt = self.name_prompt[idx]
        return img_transformed, prompt, mos
    
