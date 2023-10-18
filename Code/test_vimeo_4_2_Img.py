import torch
import torchvision
import torch.backends.cudnn as cudnn
from math import floor,sqrt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def SR_Separate(self, input):
    inputLF = input

    # img = inputLF[0,:,:,16].cpu()
    # plt.figure(2)
    # plt.imshow(img,cmap='gray')
    # plt.show()
    #
    #

    model_out_path = "video_vimeo_4_2.pth"
    checkpoint = torch.load(model_out_path)
    self.model_1.load_state_dict(checkpoint['model_1'])
    self.model_2.load_state_dict(checkpoint['model_2'])
    self.model_3.load_state_dict(checkpoint['model_3'])
    self.model_4.load_state_dict(checkpoint['model_4'])
    self.model_1.eval()
    self.model_2.eval()
    self.model_3.eval()
    self.model_4.eval()
    del checkpoint

    prediction1 = mode_test(self,inputLF)

    # # hr_img = img_HR[0, :, :, 32].cpu()
    # # plt.figure(3)
    # # plt.imshow(hr_img,cmap='gray')
    # # plt.show()
    #
    #
    # gc.collect()
    return prediction1
    # return img_HR



def mode_test(self,data):
    LF1 = self.model_1(data)
    LF1 = LF1.squeeze(1)
    LF1 = LF1.permute([0, 2, 3, 1])

    ##w,n方向
    LF2 = self.model_2(data)
    LF2 = LF2.squeeze(1)
    LF2 = LF2.permute([0, 2, 3, 1])

    ##h,n方向
    LF3 = self.model_3(data)
    LF3 = LF3.squeeze(1)
    LF3 = LF3.permute([0, 2, 3, 1])

    LF = torch.mean(torch.stack([LF1, LF2, LF3]), 0).cuda()
    del LF1,LF2,LF3

    # Enhance NEt
    prediction = self.model_4(LF)
    prediction = prediction.squeeze(1)

    return prediction


def flow_warp(data):
    _,h,w,n = data.shape
    # data = data.cpu().numpy()
    data_new = torch.zeros(1,h,w,n)
    # Img1 = torch.zeros(h,w)
    # Img2 = torch.zeros(h,w)
    # Img3 = torch.zeros(h,w)

    for i in range(n):
        if i % 2 == 0 and i !=0:
            Img1 = data[0, :, :, i - 2].cpu().numpy()
            Img2 = data[0, :, :, i - 1].cpu().numpy()
            Img3 = data[0, :, :, i].cpu().numpy()
            flow = cv2.calcOpticalFlowFarneback(Img1, Img3, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            im_warped, _ = warpImage(Img1, flow[:, :, 0], flow[:, :, 1])
            # w = int(Img1.shape[1])
            # h = int(Img1.shape[0])
            # y_coords, x_coords = np.mgrid[0:h, 0:w]
            # coords = np.float32(np.dstack([x_coords, y_coords]))
            # pixel_map = coords + flow/2
            # new_frame = cv2.remap(Img1, pixel_map, None, cv2.INTER_LINEAR)
            # new_frame = new_frame.astype(np.float32)

            Img2_new = 0.9 * im_warped+0.1*Img2
            data_new[0,:,:,i-2] = torch.from_numpy(Img1)
            data_new[0,:,:,i-1] = torch.from_numpy(Img2_new)
            data_new[0,:,:,i] = torch.from_numpy(Img3)
    data_new[0,:,:,-1] = data[0,:,:,-1]
    return data_new



