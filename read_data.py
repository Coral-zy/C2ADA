# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from pydicom import dcmread
import csv
import numpy as np
import matplotlib.pyplot as plt


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file,len_, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
                if len(image_names) > len_ : break

        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        self.data_dir = data_dir

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]


        if self.transform is not None:
            image = self.transform(image)
        #return image, torch.Tensor(label_)
        return image, torch.Tensor(label)

    def __len__(self):
        return len(self.image_names)

class Chest(Dataset):
    def __init__(self, data_dir,modal='train', transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        images = []
        labels = []
        self.data_dir = data_dir

        dir = os.path.join(data_dir, "NORMAL")
        normal = os.listdir(dir)
        for filename in normal:
            filename_loc = os.path.join(dir,filename)
            image = Image.open(filename_loc).convert('RGB')
            label = [0,1]
            images.append(image)
            labels.append(label)
            if len(images) > 100 and modal == 'train': break

        pneumonia_dir = os.path.join(data_dir, "PNEUMONIA")
        pneumonia_images = os.listdir(pneumonia_dir)
        for filename in pneumonia_images:
            filename_loc = os.path.join(pneumonia_dir, filename)
            image = Image.open(filename_loc).convert('RGB')
            label = [1,0]
            images.append(image)
            labels.append(label)
            if len(images) > 200 and modal == 'train': break



        self.transform = transform
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.Tensor(label)

    def __len__(self):
        return len(self.images)


class cad(Dataset):
    def __init__(self, data_dir,modal='train', transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        images = []
        labels = []
        self.data_dir = data_dir

        dir = os.path.join(data_dir, "mild")
        normal = os.listdir(dir)
        for filename in normal:
            filename_loc = os.path.join(dir,filename)
            image = Image.open(filename_loc).convert('RGB')
            label = [0,1]
            images.append(image)
            labels.append(label)
            if len(images) > 150 and modal=='train': break
            #if len(images) > 50 and modal == 'test': break

        pneumonia_dir = os.path.join(data_dir, "severe")
        pneumonia_images = os.listdir(pneumonia_dir)
        for filename in pneumonia_images:
            filename_loc = os.path.join(pneumonia_dir, filename)
            image = Image.open(filename_loc).convert('RGB')
            label = [1,0]
            images.append(image)
            labels.append(label)
            if len(images) > 300 and modal=='train': break
            #if len(images) > 100 and modal == 'test': break

        self.transform = transform
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.Tensor(label)

    def __len__(self):
        return len(self.images)

class RSNADataSet(Dataset):
    def __init__(self, data_dir, image_list_file,len_, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, 'r') as csv_file:
            # 创建 CSV 读取器
            csv_reader = csv.reader(csv_file)

            for row in csv_reader:
                # 在这里，row 是一个包含每一行数据的列表
                image_name = row[0]
                label = row[1]
                image_name = image_name +".dcm"
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
                if len(image_names) > len_ : break

        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        self.data_dir = data_dir

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """


        image_name = self.image_names[index]

        try:
            # 读取 DICOM 文件时的错误处理
            image = dcmread(image_name)
            image = image.pixel_array

            image_rgb = np.stack((image,) * 3, axis=-1)

            images = Image.fromarray(image_rgb)

        except Exception as e:
            print(f"读取 DICOM 文件 {image_name} 时出错：{str(e)}")
            return None

        label = self.labels[index]

        if label == 1:
            label_ = [1, 0]
        else:
            label_ = [0, 1]

        if self.transform is not None:
            # 假设 self.transform 是 torchvision 的变换
            images = self.transform(images)

        return images, torch.Tensor(label_)

    def __len__(self):
        return len(self.image_names)

