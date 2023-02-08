# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:14:11 2022

@author: Pascal Yamlome
"""



import torch
from skimage.io import imread
import pandas as pd
import nrrd
from torch.utils.data import DataLoader, Dataset

#data.Dataset

class SegmentationDataSet(Dataset):

    """
            this takes two list input image and targets that contains directories to some images and apply some 
            data transform to them
            
        """
    def __init__(self, inputs: list, targets: list, transform=None ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    #override 
    def __getitem__(self,index: int):
        #print(f'Hey index {index}')
        # Select the sample

        input_ID = self.inputs[index]
        target_ID = self.targets[index]
        
        
        #Load input and target
        x, y = nrrd.read(input_ID)[0].astype("int16") , nrrd.read(target_ID)[0]
        
        
        # #Crop images 
        # c1 = x.shape[0]//2
        # c2 = x.shape[1]//2
        
        # x = x[c1-50:c1+50, c2-50:c2+50]
        # y = y[c1-50:c1+50, c2-50:c2+50]
        
        
        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)
        
        

        
        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        
        
        return  x,y


    
# from torch.utils.data import DataLoader

# img_basedir = "G:\\Docs\\VCU\\Research\\Normative T1\\extracted_data\\ROI_T2_PAS\\"
# mask_basedir = "G:\\Docs\\VCU\\Research\\Normative T1\\extracted_data\\ROI_T2_PYG\\"
# image_files_df = pd.read_csv(img_basedir+"Mask_file_data.csv")
# msk_files_df = pd.read_csv(mask_basedir+"Mask_file_data.csv")
# inputs = []
# targets = []

# for i in range(15):
#     Base_img = img_basedir + image_files_df['Base_msk'].loc[i]
#     Base_msk = mask_basedir + msk_files_df['Base_msk'].loc[i]
    
#     inputs.append(Base_img)
#     targets.append(Base_msk)

# training_dataset = SegmentationDataSet(inputs=inputs,targets=targets,transform=None)



# # print(training_dataset.__lens__())

# print(training_dataset.__getitem__(1))