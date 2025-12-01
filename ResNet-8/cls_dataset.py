import numpy as np
import glob
import os
import sys
import re
from PIL import Image


class ClassificationJPGDataset():
    def __init__(self, imgPath):
        super(ClassificationJPGDataset, self).__init__()
        self.files_list = [] 
        self.imgPath = imgPath
        #self.files_list = sorted(glob.glob(imgPath + "/*.jpg"))
        with open(imgPath + "/y_labels.csv", 'r') as f:
            y_cont = f.read()
        f.close()
        d_parse = re.compile('(\w{1,100}.\w{1,5}),\s{0,1}\d,\s{0,1}\d')
        result_p = d_parse.findall(y_cont)
        for i in range(len(result_p)):
            self.files_list.append(result_p[i])
            self.files_list[i] = self.files_list[i].replace('bin','jpg')
        d_parse = re.compile('\w{1,50}.\w{1,5},\s{0,1}\d,\s{0,1}(\d)')
        result_p = d_parse.findall(y_cont)
        self.label_info = np.array(result_p)

    def __getitem__(self, index):
        if index < len(self.files_list):
            img = Image.open(self.imgPath + "/" + self.files_list[index]).convert('RGB')
            img_array = np.array(img)
            img_array = img_array / 255
            img_array = img_array.transpose(2, 0, 1)
            img_array = img_array.astype("float32")
            return img_array, int(self.label_info[index])
        else:
            print("Error, index={} > total_len={}".format(index, len(self.files_list)))
            return img_array, int(self.label_info[index])

    def __len__(self):
        return len(self.files_list)


class ClassificationRGBDataset():
    def __init__(self, imgPath):
        super(ClassificationRGBDataset, self).__init__()
        self.files_list = [] 
        self.imgPath = imgPath
        # self.files_list = sorted(glob.glob(imgPath + "/*.jpg"))
        with open(imgPath+"/y_labels.csv",'r') as f:
            y_cont = f.read()
        f.close()
        d_parse = re.compile('(\w{1,100}.\w{1,5}),\s{0,1}\d{1,2},\s{0,1}\d{1,2}')
        result_p = d_parse.findall(y_cont)
        for i in range(len(result_p)):
            self.files_list.append(result_p[i])
            self.files_list[i] = self.files_list[i].replace('bin','bin')
        d_parse = re.compile('\w{1,50}.\w{1,5},\s{0,1}\d{1,2},\s{0,1}(\d{1,2})')
        result_p = d_parse.findall(y_cont)
        self.label_info = np.array(result_p)
                        
    def __getitem__(self, index):
        if index < len(self.files_list):
            img = np.fromfile(self.imgPath+"/"+self.files_list[index], dtype='uint8')
            img_array = np.array(img)
            img_array = img_array.reshape(32,32,3)
            img_array = img_array.transpose(2, 0, 1)
            img_array = img_array.astype("float32")
            return img_array, int(self.label_info[index])
        else:
            print("Error, index={} > total_len={}".format(index, len(self.files_list)))
            return img_array, int(self.label_info[index])
                
    def __len__(self):
        return len(self.files_list)


class ClassificationKWSDataset():
    def __init__(self, imgPath):
        super(ClassificationKWSDataset, self).__init__()
        self.files_list = [] 
        self.imgPath = imgPath
        # self.files_list = sorted(glob.glob(imgPath + "/*.jpg"))
        with open(imgPath+"/y_labels.csv",'r') as f:
            y_cont = f.read()
        f.close()
        d_parse = re.compile('(\w{1,100}.\w{1,5}),\s{0,1}\d{1,2},\s{0,1}\d{1,2}')
        result_p = d_parse.findall(y_cont)
        for i in range(len(result_p)):
            self.files_list.append(result_p[i])
            self.files_list[i] = self.files_list[i].replace('bin','bin')
        d_parse = re.compile('\w{1,50}.\w{1,5},\s{0,1}\d{1,2},\s{0,1}(\d{1,2})')
        result_p = d_parse.findall(y_cont)
        self.label_info = np.array(result_p)
                        
    def __getitem__(self, index):
        if index < len(self.files_list):
            img = np.fromfile(self.imgPath + "/" + self.files_list[index], dtype='float32')
            img_array = np.array(img)
            img_array = img_array.reshape(49,10,1)
            img_array = img_array.transpose(2, 0, 1)
            img_array = img_array.astype("float32")
            return img_array, int(self.label_info[index])
        else:
            print("Error, index={} > total_len={}".format(index, len(self.files_list)))
            return img_array, int(self.label_info[index])
                
    def __len__(self):
        return len(self.files_list)
