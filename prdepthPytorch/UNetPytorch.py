import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

class Prior_net(torch.nn.Module):
    def __init__(self,outsize):
        super().__init__()

        # TO DO: your code here

        # 1x1 Conv layer with Downsampling
        self.conv1DS = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        #3x3 conv layer with dilation
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, dilation  = 2),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=2),
            nn.ReLU(inplace=True),
        )
        #1x1 conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        #single conv no relu
        self.conv1NR = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 1),
        )


        self.H,self.W = outsize ##41*57

    def forward(self, X):

        # TO DO: your code here
        # Conv and pools
        X = F.interpolate(X,scale_factor=(2,2),mode='bilinear',align_corners=True)
        X = self.conv1DS(X)
        X = self.conv3(X)
        X = F.interpolate(X,[self.H,self.W],mode='bilinear',align_corners=True)
        X = self.conv1(X)
        X = self.conv1NR(X)

        # X = X.view(self.H*self.W,64,1,1)
        #Attempting reshape again
        X = X.view(1,1,64,self.H*self.W)
        X = X.permute(3,2,1,0)
        # remove the last 2 dimensions
        X = torch.squeeze(X,dim =2)
        X = torch.squeeze(X,dim=2)
        #add them back
        mu = X[...,int(64/2):].unsqueeze(-1).unsqueeze(-1)
        log_sigma = X[...,:int(64/2)].unsqueeze(-1).unsqueeze(-1)
        # print(mu.shape)
        # print(log_sigma.shape)
        dist = D.Independent(D.Normal(loc=mu, scale=torch.exp(log_sigma+1e-5)),1)
        return dist


class Posterior_net(torch.nn.Module):
    def __init__(self,outsize):
        super().__init__()

        # TO DO: your code here

        #3x3 conv layer with dilation
        self.conv3DS = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size = 3, dilation  = 2),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=2),
            nn.ReLU(inplace=True),
        )
        #2x2 convs for GT depth patches each downsamples space (not channels) by 1/2
        #Each patch goes from 32X32 to 1x1 ??
        self.convGT = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size = 2, stride =2),
            nn.ReLU(inplace = True),
            nn.Conv2d(8, 16, kernel_size = 2, stride =2),
            nn.ReLU(inplace = True),
            nn.Conv2d(16, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        #single conv no relu
        self.conv1NR = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 1),
        )


        self.H,self.W = outsize ##41*57

    def forward(self, features,depth):


        # Part A (features)
        A = F.interpolate(features,[self.H+8,self.W+8],mode='bilinear',align_corners=True)
        A = self.conv3DS(A)

        #Part B(GT depth)
        depth = depth.squeeze(0)
        B = depth.permute(2, 3, 0, 1)
        B = B.view(32, 32, self.H*self.W)
        B = B.permute(2, 0, 1)
        B = B.unsqueeze(1)

        # B = depth.view(depth.shape[2]*depth.shape[1],1,32,32)
        B = self.convGT(B)
        # Attempting reshape again
        B = B.permute(2, 3, 1, 0)
        B = B.view(1,64,self.H,self.W)



        #concate and convolve
        X = torch.cat((A,B),axis=1)
        X = self.conv1(X)
        X = self.conv1NR(X)
        # X = X.view(self.H*self.W,64,1,1)
        #Attempting reshape again
        X = X.view(1,1,64,self.H*self.W)
        X = X.permute(3,2,1,0)

        # remove the last 2 dimensions
        X = torch.squeeze(X,dim =2)
        X = torch.squeeze(X,dim=2)
        # print(X.shape)

        mu = X[...,int(64/2):].unsqueeze(-1).unsqueeze(-1)
        log_sigma = X[...,:int(64/2)].unsqueeze(-1).unsqueeze(-1)
        # print(mu)
        dist = D.Independent(D.Normal(loc=mu, scale=torch.exp(log_sigma+1e-5)),1)
        return dist


class VAEModel(torch.nn.Module):
    def __init__(self,outsize):
        super().__init__()
        self.conv1DS = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        #3x3 conv layer with dilation
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, dilation  = 2),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=2),
            nn.ReLU(inplace=True),
        )
        #3x3 conv transpose
        self.conv3T =  nn.Sequential(
            nn.ConvTranspose2d(96, 96, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(96, 96, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        #2 3x3 conv transpose DS layers
        self.conv3TDS = nn.Sequential(
            nn.ConvTranspose2d(96, 64, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(64, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.outputconv = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size = 1),
            nn.Tanh(),
        )

        self.H,self.W = outsize ##41*57

        self.priornet = Prior_net(outsize).cuda()
        self.postnet = Posterior_net(outsize).cuda()
    def forward(self, X,dist,depth=None ,training=True):
        baseF = X
        X = F.interpolate(X,[self.H+8,self.W+8],mode='bilinear',align_corners=True)
        X = self.conv1DS(X)
        X = self.conv3(X)
        # X = X.view(self.H*self.W,64,1,1)
        #Attempting reshape again
        X = X.view(1,1,64,self.H*self.W)
        X = X.permute(3,2,1,0)
        #if we are training generate both samples for KL
        # if training:
        #     post_latent = self.postnet.forward(baseF,depth)
        #     prior_latent = self.priornet.forward(baseF)
        #     latent_sample = post_latent.sample()
        #     self.kl_loss = D.kl.kl_divergence(prior_latent,post_latent)
        #     self.prior_latent = prior_latent
        #     self.post_latent = post_latent
        # else:
        #     prior_latent = self.priornet.forward(baseF)
        #     latent_sample = prior_latent.sample()
        #     self.latent = prior_latent

        latent_sample = dist.sample()
        X = torch.cat((X,latent_sample),axis=1)
        X = self.conv3T(X)
        X = F.interpolate(X,scale_factor=(2,2),mode='bilinear',align_corners=True)
        X = self.conv3TDS(X)
        X = F.interpolate(X, [32,32], mode='bilinear',align_corners=True)
        out = self.outputconv(X)
        out = out.squeeze(1)
        out = out.permute(1,2,0)
        out = out.view(32,32,self.H, self.W)
        out = out.permute(2,3,0,1)
        out = out.view(self.H, self.W,32,32)
        return out



