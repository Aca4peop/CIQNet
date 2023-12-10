'''
Pre-train the model for latitude invarient representation
'''

import torch
import torch.nn as nn
from scipy import stats
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets.ODISet import ODISource,ODIData,ODIDataPro
from myresnet import resnet18


class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.conv = nn.Conv2d(64, 32,3)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.liner=nn.Linear(32,3)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.liner(x)
        return x


if __name__ == "__main__":
    # Parameters
    CVIQ_path=''
    # pre-trained models are saved in ./weights/cache/


    #
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda")
    # load data
    oidsource = ODISource()
    train_images = oidsource.train_names
    test_images = oidsource.test_names
    train_dmos = oidsource.train_mos
    test_dmos = oidsource.test_mos
    train_set = ODIDataPro(root_dir=CVIQ_path, images=train_images, dmos=train_dmos)
    test_set = ODIData(root_dir=CVIQ_path, images=test_images, dmos=test_dmos)
    dataloader = DataLoader(train_set, batch_size=8, shuffle=True, pin_memory=True)
    testloader = DataLoader(test_set, batch_size=1, shuffle=True, pin_memory=True)
    # loss
    mse = nn.MSELoss()
    CE = nn.CrossEntropyLoss()

    reformer1=nn.Conv2d(64,64,7,2,3,groups=64).half().to(device)
    reformer2 = nn.Conv2d(64, 64, 7, 2, 3, groups=64).half().to(device)
    extractor = resnet18(True).half().to(device)

    qp = nn.Sequential(nn.Linear(512,128),nn.Dropout(0.2),nn.ReLU(True),nn.Linear(128,1)).half().to(device)
    dis = nn.Sequential(nn.Linear(512,128),nn.Dropout(0.2),nn.ReLU(True),nn.Linear(128,3)).half().to(device)

    # optimizer
    Opt_reformer = torch.optim.AdamW([
        {'params': reformer1.parameters(), 'lr': 1e-5, 'eps': 1e-5},
        {'params': reformer2.parameters(), 'lr': 1e-5, 'eps': 1e-5},
    ])

    Opt_qp = torch.optim.AdamW(qp.parameters(), lr=3e-5, eps=1e-5)
    Opt_dis = torch.optim.AdamW(dis.parameters(), lr=1e-5, eps=1e-5)
    # mean and std
    me = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).half().to('cuda')
    st = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).half().to('cuda')

    ##
    for epoch in range(0,50):
        extractor.train()
        reformer1.train()
        reformer2.train()
        qp.train()
        dis.train()

        for idx, data in enumerate(tqdm(dataloader)):
            v1 = data['v1']
            v1 = v1.view(v1.shape[0] * 3, 3, 384, 1280)
            v2 = data['v2']
            v2 = v2.view(v2.shape[0] * 3, 3, 384, 1280)
            v3 = data['v3']
            v3 = v3.view(v3.shape[0] * 3, 3, 384, 1280)
            label = data['label']
            label = label.view(label.shape[0] * 3, 1)
            label = label.view(-1).half().to(device)

            # epoch<30: train qp only, epoch>30: train others
            if epoch>30:
                # GAN porcess, train Dis
                feats = torch.Tensor().half().to(device)
                labels = torch.Tensor().half().to(device)
                with torch.no_grad():
                    v = v1.half().to(device).div(255.0).sub(me).div(st)
                    feat = extractor.forward_1(v)
                    feat = reformer1(feat)
                    feat = extractor.forward_2(feat)
                    feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                feats = torch.cat((feats, feat), dim=0)
                labels = torch.cat((labels, torch.ones_like(label)*2), dim=0)

                with torch.no_grad():
                    v = v2.half().to(device).div(255.0).sub(me).div(st)
                    feat = extractor.forward_1(v)
                    feat = reformer2(feat)
                    feat = extractor.forward_2(feat)
                    feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                feats = torch.cat((feats, feat), dim=0)
                labels = torch.cat((labels, torch.ones_like(label)), dim=0)

                with torch.no_grad():
                    v = v3.half().to(device).div(255.0).sub(me).div(st)
                    feat = extractor.forward_1(v)
                    feat = extractor.forward_2(feat)
                    feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                feats = torch.cat((feats, feat), dim=0)
                labels = torch.cat((labels, torch.zeros_like(label)), dim=0)

                judge = dis(feats)
                loss = CE(judge, labels.long())
                loss.backward()
                Opt_dis.step()
                Opt_dis.zero_grad()

                #GAN porcess, train generater
                feats = torch.Tensor().half().to(device)
                labels = torch.Tensor().half().to(device)

                v = v1.half().to(device).div(255.0).sub(me).div(st)
                feat = extractor.forward_1(v)
                feat = reformer1(feat)
                feat = extractor.forward_2(feat)
                feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                feats = torch.cat((feats, feat), dim=0)
                labels = torch.cat((labels, label), dim=0)

                v = v2.half().to(device).div(255.0).sub(me).div(st)
                feat = extractor.forward_1(v)
                feat = reformer2(feat)
                feat = extractor.forward_2(feat)
                feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                feats = torch.cat((feats, feat), dim=0)
                labels = torch.cat((labels, label), dim=0)

                predictions = qp(feats).view(-1)
                loss_mse = mse(predictions, labels)

                loss = loss_mse
                loss.backward()
                Opt_reformer.step()
                Opt_reformer.zero_grad()
                dis.zero_grad()
                qp.zero_grad()
            else:
                #train qp only
                v = v3.half().to(device).div(255.0).sub(me).div(st)
                with torch.no_grad():
                    feat = extractor(v)
                feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                predictions = qp(feat).view(-1)
                loss_mse = mse(predictions, label)
                loss_mse.backward()
                Opt_qp.step()
                Opt_qp.zero_grad()

        # <-----------------eval ---------------------->
        srocclast = 0
        extractor.eval()
        reformer1.eval()
        reformer2.eval()
        qp.eval()
        dis.eval()
        #predictions of 5 sub-areas
        pre1 = np.array([0])
        pre2 = np.array([0])
        pre3 = np.array([0])
        pre4 = np.array([0])
        pre5 = np.array([0])
        pre = [pre1, pre2, pre3,pre4,pre5]
        tar = np.array([0])
        correct = 0
        with torch.no_grad():
            for idx, data in enumerate(tqdm(testloader)):
                v1 = data['v1']
                v2 = data['v2']
                v3 = data['v3']
                v4 = data['v4']
                v5 = data['v5']
                label = data['label']
                del data
                feats = torch.Tensor().half().to(device)
                classes = torch.Tensor().half().to(device)
                for k, v in enumerate([v1, v2, v3, v4, v5]):
                    v = v.half().to(device).div(255.0).sub(me).div(st)
                    with torch.no_grad():
                        if k == 2:# Equatorial area
                            feat = extractor(v)
                            feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                            feats = torch.cat((feats, feat), dim=0)
                            classes = torch.cat((classes, torch.zeros_like(label, device=device)), dim=0)
                        elif k in [0,4]:
                            feat = extractor.forward_1(v)
                            feat = reformer1(feat)
                            feat = extractor.forward_2(feat)
                            feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                            feats = torch.cat((feats, feat), dim=0)
                            classes = torch.cat((classes, torch.ones_like(label, device=device)*2), dim=0)
                        else:
                            feat = extractor.forward_1(v)
                            feat = reformer2(feat)
                            feat = extractor.forward_2(feat)
                            feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                            feats = torch.cat((feats, feat), dim=0)
                            classes = torch.cat((classes, torch.ones_like(label, device=device) ), dim=0)

                    predictions = qp(feat).view(-1)
                    output = predictions.to('cpu')
                    predict = output.data.numpy().flatten()
                    pre[k] = np.hstack((pre[k], np.mean(predict)))
                tar = np.hstack((tar, label))
                judge = dis(feats)
                predicted = torch.max(judge.data, 1)[1]
                correct += (predicted == classes).sum()
            srocc1, _ = stats.spearmanr(pre[0][1:], tar[1:])
            srocc2, _ = stats.spearmanr(pre[1][1:], tar[1:])
            srocc3, _ = stats.spearmanr(pre[2][1:], tar[1:])
            srocc4, _ = stats.spearmanr(pre[3][1:], tar[1:])
            srocc5, _ = stats.spearmanr(pre[4][1:], tar[1:])
            correct = correct / len(testloader) / 5
            print('epc :%d ,srcc3 :%.4f,srcc4 :%.4f,srcc5 :%.4f, ACC:%.4f' % (epoch,  srocc3,srocc4, srocc5, correct))

            checkpoint = {'reformer1':reformer1.state_dict(),'reformer2':reformer2.state_dict()}
            if epoch>30:
                torch.save(checkpoint, './weights/cache/CVIQv5_epoch%d_qp%.4f_dis%.4f_reformer%.4f_reformer%.4f.pth' % (epoch,srocc2,correct,srocc4,srocc5))
