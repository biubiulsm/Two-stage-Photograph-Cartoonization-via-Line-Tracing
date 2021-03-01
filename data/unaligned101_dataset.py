import os.path
from data.base_dataset import BaseDataset, get_transform, get_transform2
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
class Unaligned101Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_BS = os.path.join(opt.dataroot, opt.phase + 'BS')
        self.dir_AE = os.path.join(opt.dataroot, opt.phase + 'AE')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.BS_paths = make_dataset(self.dir_BS)
        self.AE_paths = make_dataset(self.dir_AE)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.BS_paths = sorted(self.BS_paths)
        self.AE_paths = sorted(self.AE_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.BS_size = len(self.BS_paths)
        self.AE_size = len(self.AE_paths)

        self.transform = get_transform(opt)
        self.transform2 = get_transform2(opt)
        
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        AE_path = self.AE_paths[index % self.AE_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
            index_B2 = index % self.B_size
            index_B3 = index % self.B_size
            index_B4 = index % self.B_size
            index_B5 = index % self.B_size

            index_BS = index % self.BS_size
            index_BS2 = index % self.BS_size
            index_BS3 = index % self.BS_size
            index_BS4 = index % self.BS_size
            index_BS5 = index % self.BS_size             
        else:
            index_B = random.randint(0, self.B_size - 1)
            index_B2 = random.randint(0, self.B_size - 1)
            index_B3 = random.randint(0, self.B_size - 1)
            index_B4 = random.randint(0, self.B_size - 1)
            index_B5 = random.randint(0, self.B_size - 1)

            index_BS = random.randint(0, self.BS_size - 1)
            index_BS2 = random.randint(0, self.BS_size - 1)
            index_BS3 = random.randint(0, self.BS_size - 1)
            index_BS4 = random.randint(0, self.BS_size - 1)
            index_BS5 = random.randint(0, self.BS_size - 1)

        B_path = self.B_paths[index_B]
        B2_path = self.B_paths[index_B2]
        B3_path = self.B_paths[index_B3]
        B4_path = self.B_paths[index_B4]
        B5_path = self.B_paths[index_B5]

        # BS_path = self.BS_paths[index_BS]
        # BS2_path = self.BS_paths[index_BS2]
        # BS3_path = self.BS_paths[index_BS3]
        # BS4_path = self.BS_paths[index_BS4]
        # BS5_path = self.BS_paths[index_BS5]

        BS_path = self.BS_paths[index_B]
        BS2_path = self.BS_paths[index_B2]
        BS3_path = self.BS_paths[index_B3]
        BS4_path = self.BS_paths[index_B4]
        BS5_path = self.BS_paths[index_B5]

        A_img = Image.open(A_path).convert('RGB')
        AE_img = Image.open(AE_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        B2_img = Image.open(B2_path).convert('RGB')
        B3_img = Image.open(B3_path).convert('RGB')
        B4_img = Image.open(B4_path).convert('RGB')
        B5_img = Image.open(B5_path).convert('RGB')

        BS_img = Image.open(BS_path).convert('RGB')
        BS2_img = Image.open(BS2_path).convert('RGB')
        BS3_img = Image.open(BS3_path).convert('RGB')
        BS4_img = Image.open(BS4_path).convert('RGB')
        BS5_img = Image.open(BS5_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform2(B_img)
        B2 = self.transform2(B2_img)
        B3 = self.transform2(B3_img)
        B4 = self.transform2(B4_img)
        B5 = self.transform2(B5_img)

        BS = self.transform2(BS_img)
        BS2 = self.transform2(BS2_img)
        BS3 = self.transform2(BS3_img)
        BS4 = self.transform2(BS4_img)
        BS5 = self.transform2(BS5_img)

        AE_img = np.array(AE_img)
        # print(BS_img.shape)
        v = [200, 200, 200]
        AE_img[AE_img>=v] = 255
        AE_img[AE_img<v] = 0
        AE = self.transform2(Image.fromarray(AE_img))
        tmp = AE[0, ...] * 0.299 + AE[1, ...] * 0.587 + AE[2, ...] * 0.114
        AE = tmp.unsqueeze(0)
        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        
        if output_nc == 1:  # RGB to gray
            tmp = B2[0, ...] * 0.299 + B2[1, ...] * 0.587 + B2[2, ...] * 0.114
            B2 = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B3[0, ...] * 0.299 + B3[1, ...] * 0.587 + B3[2, ...] * 0.114
            B3 = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B4[0, ...] * 0.299 + B4[1, ...] * 0.587 + B4[2, ...] * 0.114
            B4 = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B5[0, ...] * 0.299 + B5[1, ...] * 0.587 + B5[2, ...] * 0.114
            B5 = tmp.unsqueeze(0)

        # BS
        if output_nc == 1:  # RGB to gray
            tmp = BS[0, ...] * 0.299 + BS[1, ...] * 0.587 + BS[2, ...] * 0.114
            BS = tmp.unsqueeze(0)
        
        if output_nc == 1:  # RGB to gray
            tmp = BS2[0, ...] * 0.299 + BS2[1, ...] * 0.587 + BS2[2, ...] * 0.114
            BS2 = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = BS3[0, ...] * 0.299 + BS3[1, ...] * 0.587 + BS3[2, ...] * 0.114
            BS3 = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = BS4[0, ...] * 0.299 + BS4[1, ...] * 0.587 + BS4[2, ...] * 0.114
            BS4 = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = BS5[0, ...] * 0.299 + BS5[1, ...] * 0.587 + BS5[2, ...] * 0.114
            BS5 = tmp.unsqueeze(0)

        return {'A': A, 'AE': AE, 'B': B, 'B2': B2, 'B3': B3, 'B4': B4, 'B5': B5,
                'BS': BS, 'BS2': BS2, 'BS3': BS3, 'BS4': BS4, 'BS5': BS5,
                'A_paths': A_path, 'B_paths': B_path, 'B2_paths': B2_path,
                'B3_paths': B3_path, 'B4_paths': B4_path, 'B5_paths': B5_path,
                'BS_paths': BS_path, 'BS2_paths': BS2_path,'BS3_paths': BS3_path, 
                'BS4_paths': BS4_path, 'BS5_paths': BS5_path}

    def __len__(self):
        return max(self.A_size, self.B_size, self.BS_size)

    def name(self):
        return 'Unaligned101Dataset'
