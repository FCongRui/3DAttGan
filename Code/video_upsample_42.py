
import torch
import torch.nn as nn


def make_RDN_model_upsample(args,flag):
    return RDN(args,flag)
def make_ARCNN_model_upsample():
    return ARCNN()


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize -1 )//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c* G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self, args,flag):
        super(RDN, self).__init__()
        r = args.upscale_factor
        G0 = args.G0
        kSize = args.RDNkSize

        self.flag = flag

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
            'C': (8, 6, 64),
        }[args.RDNconfig]

       #######32_64#######
        if self.flag =='SAI':
            a = tuple([4, 4])
            self.K, S, P, OP = (3, [2, 1, 1], 1,0)

        if self.flag =='EPI_1':
            a = tuple([4, 1.75])
            self.K, S, P, OP = ([3,5,3], [1, 4, 1], 1,[ 0, 1, 0])

        if self.flag == 'EPI_2':
            a = tuple([4, 1.75])
            self.K, S, P, OP = ([3,3,5], [1, 1, 4], 1, [0, 0, 1])

        self.upsample = nn.Upsample(scale_factor= a,mode='bicubic') #height = weigh
        # self.upsample = nn.Upsample(scale_factor=args.upscale_factor, mode='bicubic')  # height = weigh
        # self.upsample = nn.Upsample(size=(a,b), mode='bicubic')   #heigh！=weigh

        # Shallow feature extraction ne
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])


        self.Conv = nn.Conv2d(G, 3, kSize, padding=(kSize - 1) // 2, stride=1)


        self.upFrame = nn.Sequential(
            nn.ConvTranspose3d(3,G,kernel_size=self.K,stride=S,padding=P,output_padding=OP),
            nn.LeakyReLU(0.2),
            nn.Conv3d(G,args.n_colors,kernel_size= 3,stride=1,padding=1)
        )


    def forward(self, data):

        b, h, w, c,n = data.shape

        if self.flag == 'SAI':
           y1 = torch.zeros(b, 3, h * 4, w * 4, n).cuda()
           for j in range(n):
               x = data[:, :, :,:, j] #b,h,w,c,n


               x = x.permute([0,3,1,2])  # b,c,h,w

               x_up = self.upsample(x)
               # x = F.interpolate(x,scale_factor=2,mode='bicubic')
               f__1 = self.SFENet1(x_up)
               x = self.SFENet2(f__1)

               RDBs_out = []
               for i in range(self.D):
                   x = self.RDBs[i](x)
                   RDBs_out.append(x)

               x = self.GFF(torch.cat(RDBs_out, 1))

               x = self.Conv(x)

               x = x_up +x

               # x = self.Conv(x)

               y1[:, :, :, :, j] = x   #b,c,h,w,n
           y1 = y1.permute([0,1,4,2,3])  #b,c,n,h,w
           y = self.upFrame(y1)    ##帧率提升
           del y1
           return  y

        if self.flag == 'EPI_1':
            y2 = torch.zeros(b, 3, h, w * 4, 7).cuda()
            for j in range(h):
                x = data[:, j, :, :,:]  # w,n 共 h个  #b,h,w,c,n
                # x = x.unsqueeze(1)
                x = x.permute([0, 2, 1, 3])  # b,c,w,n

                x_up = self.upsample(x)
                # x = F.interpolate(x,scale_factor=2,mode='bicubic')
                f__1 = self.SFENet1(x_up)
                x = self.SFENet2(f__1)

                RDBs_out = []
                for i in range(self.D):
                    x = self.RDBs[i](x)
                    RDBs_out.append(x)

                x = self.GFF(torch.cat(RDBs_out, 1))

                x = self.Conv(x)

                x = x_up +x

                y2[:, :, j, :, :] = x  #b,c,h,w,n
            y2 = y2.permute([0, 1, 4, 2, 3])  #b,c,n,h,w
            y = self.upFrame(y2)  ##帧率提升
            del y2
            return y

        if self.flag == 'EPI_2':
            y3 = torch.zeros(b, 3, h * 4, w, 7).cuda()
            for j in range(w):
                x = data[:, :, j, :,:]  # h,n共w个 #b,h,c,n
                x = x.permute([0, 2, 1, 3])  # b,c,h,n
                x_up = self.upsample(x)
                # x = F.interpolate(x,scale_factor=2,mode='bicubic')
                f__1 = self.SFENet1(x_up)
                x = self.SFENet2(f__1)

                RDBs_out = []
                for i in range(self.D):
                    x = self.RDBs[i](x)
                    RDBs_out.append(x)

                x = self.GFF(torch.cat(RDBs_out, 1))

                x = self.Conv(x)

                x = x_up +x

                y3[:, :, :, j, :] = x  # ah,w,an
            y3 = y3.permute([0, 1, 4, 2, 3])
            y = self.upFrame(y3)  ##帧率提升
            del y3
            return y

class ARCNN(nn.Module):
    def __init__(self):
        super(ARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.PReLU()
        )
        self.last = nn.Conv2d(16, 3, kernel_size=5, padding=2)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, data):

        b, h, w,c, a = data.shape
        out = torch.zeros(b, 3, h, w, a).cuda()
        for i in range(a):
            x = data[:, :, :, :,i]  # b,h,w,c
            x = x.permute([0, 3, 1, 2])  # b,c,h,w
            x = self.base(x)
            x = self.last(x)
            out[:, :, :, :, i] = x

        return out



