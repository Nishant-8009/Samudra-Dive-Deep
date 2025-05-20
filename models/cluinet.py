import torch
import os
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.utils import save_image
from torchvision import transforms


class testdataset(Dataset):
    def __init__(self, data_path, img_format='jpg'):
        self.data_path = data_path
        self.uw_images = glob(os.path.join(self.data_path, '*.' + img_format))
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, index):
            uw_img = self.transform(Image.open(self.uw_images[index]))
            name = os.path.basename(self.uw_images[index])
            return uw_img,name
    def __len__(self):
        return len(self.uw_images)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):   
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):  
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module): 
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]


        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):  
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
        
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UNetEncoder(nn.Module):
    def __init__(self, n_channels=3):
        super(UNetEncoder, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5, (x1, x2, x3, x4)

class UNetDecoder(nn.Module):
    def __init__(self, n_channels=3):
        super(UNetDecoder, self).__init__()
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, enc_outs):
        x = self.sigmoid(x)
        x = self.up1(x, enc_outs[3])
        x = self.up2(x, enc_outs[2])
        x = self.up3(x, enc_outs[1])
        x = self.up4(x, enc_outs[0])
        x = self.outc(x)
        return nn.Tanh()(x)

def write_to_log(log_file_path, status):
    with open(log_file_path, "a") as log_file:
        log_file.write(status + '\n')

def to_img(x,wide,hig):
    """Convert the tanh (-1 to 1) ranged tensor to image (0 to 1) tensor	"""
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, wide, hig)
    return x


def output(fE, fI, dataloader,output_path):
    fE.eval()
    fI.eval()
    for idx, data in tqdm(enumerate(dataloader)):
        inputtestimg, testname = data
        inputtestimg = Variable(inputtestimg).cuda()
        fE_out, enc_outs = fE(inputtestimg)
        fI_out = to_img(fI(fE_out, enc_outs),inputtestimg.shape[2],inputtestimg.shape[3])
        save_image(fI_out.cpu().data, output_path + '/{}'.format(testname[0]))


# name='UIEBD'
# test_path ='/home/cvpr/Downloads/CLUIE-Net-CLUIE-Net/data/test_demo'
# fe_load_path='/home/cvpr/Downloads/CLUIE-Net-CLUIE-Net/fE_latest.pth'
# fi_load_path='/home/cvpr/Downloads/CLUIE-Net-CLUIE-Net/fI_latest.pth'
# output_path='/home/cvpr/Downloads/CLUIE-Net-CLUIE-Net/output'

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# fE_load_path = fe_load_path
# fI_load_path = fi_load_path
# test_dataset = testdataset(test_path)
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# fE = UNetEncoder().cuda()
# fI = UNetDecoder().cuda()
# fE.load_state_dict(torch.load(fE_load_path))
# fI.load_state_dict(torch.load(fI_load_path))
# output(fE, fI, test_dataloader,output_path)

