import torch
import torch.nn as nn
import numpy as np
from math import log10,sqrt
import cv2 as cv
from traintestfunction import SR_Separate,SuperResolution_Model
from Enhance_inter import Enhance_Block

from network_3d import make_3d_network
import torch.backends.cudnn as cudnn
import gc
import os
from scipy.io import loadmat,savemat
from sewar.full_ref import ssim,psnr
# import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Trainer(object):
    def __init__(self, config, training_loader,testing_loader):
        super(Trainer, self).__init__()
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None      #lr 下降方式的控制
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.fig = config.fig

    def build_model(self):
        self.model = make_3d_network()

        # if torch.cuda.device_count()>1:
        #     print("let's use", torch.cuda.device_count(), "GPUs!")
        #     self.model = nn.DataParallel(self.model)

        self.model.to(self.device)


        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},

        ],
            # filter(lambda p: p.requires_grad, self.model_5.parameters()),
            lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40, 70, 90, 100], gamma=0.5) # lr decay
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)


    def save(self,epoch,i):
        # model_out_path = "Super_video_vimeoAll_4_2.pth"
        model_out_path1 = "epoch_lr_data_RGB_video.pth"
        model_out_path2 = "batch_lr_data_RGB_video.pth"
        state = {
                 'model': self.model.state_dict(),
                 'epoch' : epoch
                 }
        if i==0:
            torch.save(state, model_out_path2)
        else:
            torch.save(state, model_out_path1)

        # torch.save(self.model, model_out_path)
        # print("Checkpoint saved to {}".format(model_out_path2))

    def train(self,epoch):

        self.model.train()

        # Maxbatchnum = 3000

        f = open("Loss_lr_data_RGB_video.txt", "a")

        train_loss = 0

        for batch_num, (data, target) in enumerate(self.training_loader):
            # if batch_num >= Maxbatchnum:
            #     break
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            f1 = open("batch_Loss_lr_data_RGB_video.txt", "a")

            try:
                #loss_1,loss_3 = SuperResolution_Model(self,data,target)
                # loss_5 = SuperResolution_Model(self, data, target)
                # loss = 0.7*loss_1+0.3*loss_2+0.3*loss_3
                target = target.permute([0, 3, 4, 1, 2])
                data = data.permute([0, 3,4,1,2])

                predict = self.model(data)
                loss = self.criterion(predict, target)
                #loss = loss_1+0.3*loss_3
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            # loss_1,loss_2,loss_3,loss_4 = SuperResolution_Model(self,data,target)
            # loss_1, loss_2, loss_3, loss_4, loss_5 = SuperResolution_Model(self, data, target)

            # loss = loss_1+loss_2+loss_3+loss_4
            # train_loss += loss.item()
            # loss.backward()
            # self.optimizer.step()

            if batch_num % 10 == 0:
                # print(" Epoch:{:d}\tbatch_num:{:d}\tLoss: {:.4f}\t".format(epoch, batch_num, loss.item()))
                print(" Epoch:{:d}\tbatch_num:{:d}\tLoss: {:.4f}\t".format(epoch, batch_num, train_loss / (batch_num + 1)))
                f1.write("Epoch:{:d}, batch_num:{:d} Loss: {:.4f}\n".format(epoch, batch_num, train_loss / (batch_num + 1)))
                self.save(epoch,0)
                f1.close()

        # print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))
        print("=========Epoch:{:d}\t Average Loss: {:.4f}".format(epoch,train_loss/ (batch_num + 1)))
        f.write("Epoch:{:d},  Average Loss: {:.4f}\n".format(epoch,train_loss / (batch_num + 1)))
        f.close()


    def video_test(self):
            f = open("PSNR_SSIM_lr_data_RGB_video_Avg_slow.txt", "a")
            f1 = open("PSNR_SSIM_lr_data_RGB_video_SR_slow.txt", "a")
            f2 = open("PSNR_SSIM_lr_data_RGB_video_VFI_slow.txt", "a")
            f3 = open("PSNR_SSIM_lr_data_RGB_videoo_Alldetil_slow.txt", "a")

            avg_psnr = 0
            avg_ssim = 0
            avg_SR_psnr = 0
            avg_SR_ssim = 0
            avg_VFI_psnr = 0
            avg_VFI_ssim = 0
            # a = 6
            a = 7
            with torch.no_grad():
                for batch_num, (data, target,Imgname) in enumerate(self.testing_loader):
                    print("####################{:d}############".format(batch_num))
                    data, target = data.to(self.device), target.to(self.device)

                    file = Imgname[0]
                    print(file)
                    # savepath = "./Unet_RGB_Save_video_vimeo_slow/" + file + "/"
                    # isExists = os.path.exists(savepath)
                    # if not isExists:
                    #     os.makedirs(savepath)

                    # target = target.permute([0, 3, 4, 1, 2])
                    data = data.permute([0, 3, 4, 1, 2])

                    predict = self.model(data)  #b,c,n,h,w
                    predict = predict.permute([0, 3, 4, 1, 2])  #b,h,w,c,n


                    # 保存测试结果为.mat
                    # savepathmat = "./Save_video_vimeo_mat/"
                    # isExists = os.path.exists(savepathmat)
                    # if not isExists:
                    #     os.makedirs(savepathmat)
                    # filemat = Imgname[0][-14:-4]
                    # dataNew = savepathmat + filemat + ".mat"
                    # savemat(dataNew, {'Input': prediction.cpu().numpy(),'Target': target.cpu().numpy()})

                    psnr_score = np.zeros(a)
                    curSSIM = np.zeros(a)
                    SR_psnr = 0
                    SR_ssim = 0
                    VFI_psnr = 0
                    VFI_ssim = 0
                    number = 4
                    f3.write(file)
                    # f3.write("\n")


                    for i in range(a):

                        # 保存测试结果为图片
                        # path = savepath + str(i)+".png"
                        # img_R = prediction[0, :, :, 0:1, i]*255
                        # img_G = prediction[0, :, :, 1:2, i] * 255
                        # img_B = prediction[0, :, :, 2::, i] * 255
                        # img = torch.cat([img_B,img_G,img_R],2)
                        # img = np.array(img.cpu().numpy(), dtype='uint8')
                        # cv.imwrite(path, img)   # save to .png

                        img1 = predict[0, :, :, :,i]
                        img2 = target[0, :, :,:, i]
                        # cv.imwrite(savepath + "t.png", img2.cpu().numpy())


                        # psnr_score[i] = Cp_PSNR(prediction[0, :, :, i], target[0, :, :, i])

                        psnr_score[i] = psnr(img2.cpu().numpy(), img1.cpu().numpy(), MAX=1)
                        curSSIM[i] = ssim(img2.cpu().numpy(), img1.cpu().numpy(), MAX=1)[0]

                        print("psnr:{:.4f}".format(psnr_score[i]))
                        print("ssim:{:.4f}".format(curSSIM[i]))
                        f3.write(" PSNR: {:.4f}\t".format(psnr_score[i]))
                        f3.write(" SSIM: {:.4f}\t".format(curSSIM[i]))
                        f3.write("\n")

                        if i%2==0:
                            SR_psnr = SR_psnr +  psnr_score[i]
                            SR_ssim = SR_ssim +  curSSIM[i]
                        else:
                            VFI_psnr = VFI_psnr +psnr_score[i]
                            VFI_ssim = VFI_ssim + curSSIM[i]

                    PSNR_SR = SR_psnr/4
                    PSNR_VFI = VFI_psnr / 3
                    # PSNR_VFI = VFI_psnr/3
                    PSNR = np.mean(psnr_score)
                    print(" each one_Img PSNR: {:.4f}".format(PSNR))
                    print(" SR PSNR: {:.4f}".format(PSNR_SR))
                    print(" VFI PSNR: {:.4f}".format(PSNR_VFI))
                    f.write(file)
                    f.write(" PSNR: {:.4f}\t".format(PSNR))
                    f1.write(file)
                    f1.write(" SR PSNR: {:.4f}\t".format(PSNR_SR))
                    f2.write(file)
                    f2.write(" VFI PSNR: {:.4f}\t".format(PSNR_VFI))
                    avg_psnr += PSNR
                    avg_SR_psnr +=PSNR_SR
                    avg_VFI_psnr +=PSNR_VFI

                    SSIM = np.mean(curSSIM)
                    SSIM_SR = SR_ssim / 4
                    SSIM_VFI = VFI_ssim / 3
                    # SSIM_VFI = VFI_ssim / 3
                    print(" each one_Img SSIM: {:.4f}".format(SSIM))
                    print(" SR SSIM: {:.4f}".format(SSIM_SR))
                    print(" VFI SSIM: {:.4f}".format(SSIM_VFI))
                    # f.write(file)
                    f.write(" SSIM: {:.4f}\n".format(SSIM))
                    # f1.write(file)
                    f1.write(" SR SSIM: {:.4f}\n".format(SSIM_SR))
                    # f2.write(file)
                    f2.write(" VFI SSIM: {:.4f}\n".format(SSIM_VFI))
                    avg_ssim += SSIM
                    avg_SR_ssim +=SSIM_SR
                    avg_VFI_ssim +=SSIM_VFI

                    # del data, target, prediction
                    gc.collect()

                print("=========Average Psnr: {:.4f}".format(avg_psnr / len(self.testing_loader)))
                print("=========Average SR_Psnr: {:.4f}".format(avg_SR_psnr / len(self.testing_loader)))
                print("=========Average VFI_Psnr: {:.4f}".format(avg_VFI_psnr / len(self.testing_loader)))
                f.write("=========Average Psnr: {:.4f}\n".format(avg_psnr / len(self.testing_loader)))
                f1.write("=========Average SR_Psnr: {:.4f}\n".format(avg_SR_psnr / len(self.testing_loader)))
                f2.write("=========Average VFI_Psnr: {:.4f}\n".format(avg_VFI_psnr / len(self.testing_loader)))

                print("=========Average SSIM: {:.4f}".format(avg_ssim / len(self.testing_loader)))
                print("=========Average SR_SSIM: {:.4f}".format(avg_SR_ssim / len(self.testing_loader)))
                print("=========Average VFI_SSIM: {:.4f}".format(avg_VFI_ssim / len(self.testing_loader)))
                f.write("=========Average SSIM: {:.4f}\n".format(avg_ssim / len(self.testing_loader)))
                f1.write("=========Average SR_SSIM: {:.4f}\n".format(avg_SR_ssim / len(self.testing_loader)))
                f2.write("=========Average VFI_SSIM: {:.4f}\n".format(avg_VFI_ssim / len(self.testing_loader)))
            f.close()
            f1.close()
            f2.close()
            f3.close()


    def run(self,args):

        if self.fig =='train':
            self.build_model(args)

            # model_out_path = "epoch_Unet_RGB_video_vimeoAll_4_2.pth"
            # checkpoint = torch.load(model_out_path)
            # self.model_1.load_state_dict(checkpoint['model_1'])
            # self.model_2.load_state_dict(checkpoint['model_2'])
            # self.model_3.load_state_dict(checkpoint['model_3'])
            # self.model_4.load_state_dict(checkpoint['model_4'])
            # self.model_5.load_state_dict(checkpoint['model_5'])
            # start_epoch = checkpoint['epoch'] + 1

            start_epoch =1
            for epoch in range(start_epoch, self.nEpochs + 1):
                print("\n===> Epoch {} starts:".format(epoch))

                self.train(epoch)
                self.scheduler.step(epoch)
                if epoch % 1 == 0:
                    self.save(epoch,1)
                    print("Checkpoint saved" )
        if self.fig == 'test':
            self.build_model(args)
            model_out_path = "epoch_lr_data_RGB_video.pth"
            # model_out_path2 = "Enhanced_video_vimeoAll_4_2.pth"
            checkpoint = torch.load(model_out_path)
            # checkpoint2 = torch.load(model_out_path2)
            self.model.load_state_dict(checkpoint['model'])

            self.model.eval()

            self.video_test()

# def Cp_PSNR(img1,img2):
#
#
#     diff = img1 - img2
#
#     # plt.figure(3)
#     # plt.imshow(diff.cpu(), cmap='gray')
#
#     diff = diff.flatten()
#     rmse = sqrt(torch.mean(diff ** 2.))
#     return 20 * log10(1.0 / rmse)
#
#     # mse = np.mean((img1  - img2 ) ** 2)
#     # if mse < 1.0e-10:
#     #     return 100
#     # PIXEL_MAX = 1
#     # return 20 * log10(PIXEL_MAX / sqrt(mse))
