'''
code for intradataset evaluation

'''


import os.path
import torch
from torch.optim import Adam, AdamW
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import stats

from CasualOVQA import CasualVQA
from datasets.DataSource import ODVSource,JVQDSource,SVQDSource

class VideoFeat(Dataset):
    def __init__(self, images, dmos,fpath, vlengh):
        self.images = images
        self.dmos = dmos
        self.vlen=vlengh
        self.fpath=fpath

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        vname = self.images[index]
        feat=np.load(os.path.join(self.fpath,vname[:-4]+'.npy'))
        features=np.zeros((self.vlen, 5,512))
        features[0:self.vlen,:]=feat[0:self.vlen,:]
        dmos=self.dmos[index]
        sample = {'feat': features, 'label': dmos,'name':vname}
        return sample


if __name__ == "__main__":
    # parameters
    Videosource=SVQDSource
    feature_path='./features/SVQD'
    video_length=500 #ODV: 300  JVQD:120 SVQD: 500
    
    #
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #
    device = torch.device("cuda")
    
    videosource = Videosource() # generate tran-test splits
    bsrcc=np.zeros((5,1))
    bplcc = np.zeros((5, 1))
    brmse = np.zeros((5, 1))

    for r in range(0,5):
        rounds_index = r
        checkpoint = videosource.fiveFolds[r] # 5-folds eval
        train_images = checkpoint['train_images']
        train_dmos = checkpoint['train_dmos']
        test_images = checkpoint['test_images']
        test_dmos = checkpoint['test_dmos']

        train_set = VideoFeat( images=train_images, dmos=train_dmos,fpath=feature_path,vlengh=video_length)
        test_set = VideoFeat( images=test_images, dmos=test_dmos,fpath=feature_path,vlengh=video_length)
        dataloader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2,pin_memory=True)
        testloader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2,pin_memory=True)

        model = CasualVQA().float().to(device)
        criterion = nn.MSELoss()
        optimizer = AdamW(model.parameters(), lr=3e-5)
        sroccbest=0

        for epoch in range(300):
            # ------train-------------
            model.train()
            L = 0
            for idx, data in enumerate((dataloader)):
                features=data['feat']
                label=data['label']
                features = features.to(device).float()
                label = label.to(device).float()
                optimizer.zero_grad()  #
                outputs = model(features)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                L = L + loss.item()
            train_loss = L / (idx + 1)

            # ------eval-------------
            model.eval()
            pre = np.array([0])
            tar = np.array([0])
            mydic=dict()
            for idx, data in enumerate((testloader)):
                features=data['feat']
                label=data['label']
                features = features.to(device).float()
                label = label.data.numpy().flatten()
                output = model(features)
                output = output.to('cpu')
                predict = output.data.numpy().flatten()
                pre = np.hstack((pre, predict))
                tar = np.hstack((tar, label))

            srocc1, _ = stats.spearmanr(pre[1:], tar[1:])
            plcc1, _ = stats.pearsonr(pre[1:], tar[1:])
            rmse1 = np.sqrt(np.mean(np.square(pre[1:] - tar[1:])))

            if srocc1>sroccbest:
                sroccbest=srocc1
                bsrcc[r]=srocc1
                bplcc[r]=plcc1
                brmse[r]=rmse1
                checkpoint={"model_state_dict": model.state_dict()}
                # torch.save(checkpoint,'./model/model_'+str(rounds_index)+'.pth')
                print('epoch %d, plcc %.4f , srocc %.4f ,rmse %.4f,%.4f' % (r,plcc1, srocc1, rmse1,sroccbest))

    print(np.mean(bsrcc))
    print(np.mean(bplcc))
    print(np.mean(brmse))