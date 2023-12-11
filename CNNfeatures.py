# Extract fremlevel features from videos

import torch
import torch.nn as nn
import skvideo.io
import os
import numpy as np
from tqdm import tqdm
import skvideo.io
from myresnet import resnet18



if __name__ == "__main__":
    # parameters
    video_path = '/home2/SVQD/'
    feature_path = 'features/SVQD'
    video_length=500  #ODV:300  JVQD:120  SVQD:500
    weight_path= 'weights/pretrained/CVIQv5.pth'


    device = torch.device("cuda")

    # Instantiating the backbone
    extractor1 = resnet18(True).half().to(device)
    extractor1.eval()
    # We add a 7×7 convolutional layer after the first convolutional layer of the feature encoder to adjust the feature maps
    reformer1 = nn.Conv2d(64,64,7,2,3,groups=64).half().to(device)
    reformer2 = nn.Conv2d(64, 64, 7, 2, 3, groups=64).half().to(device)
    reformer1.eval()
    reformer2.eval()

    # Load the weights
    checkpoint = torch.load(weight_path)
    reformer1.load_state_dict(checkpoint['reformer1'])
    reformer2.load_state_dict(checkpoint['reformer2'])
    
    # Mean and std for the normalization
    me = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).half().to('cuda')
    st = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).half().to('cuda')
    
    with torch.no_grad():


        videos=os.listdir(video_path)
        
        for vname in tqdm(videos):
            if not ('.mp4' in vname or '.mov' in vname or '.mkv' in vname):
                continue
            ffmpeg=skvideo.io.FFmpegReader(os.path.join(video_path,vname),outputdict={'-s':'3840x1920'})
            features = torch.Tensor().to(device)
            fcount = 0

            for frame in ffmpeg.nextFrame():
                frame=np.transpose(frame,[2,0,1]) # (H,W,C) -> (C,H,W)
                frame=torch.from_numpy(frame).half()
                frame = frame.to(device).div(255.0).sub(me).div(st)
                v1 = frame[:, :, 0:384, :]
                v2 = frame[:, :, 384:384 * 2, :]
                v3 = frame[:, :, 384 * 2:384 * 3, :]
                v4 = frame[:, :, 384 * 3:384 * 4, :]
                v5 = frame[:, :, 384 * 4:, :]


                v1=extractor1.forward_1(v1)
                v1=reformer1(v1)
                v1=extractor1.forward_2(v1)
                v1=nn.functional.adaptive_avg_pool2d(v1, 1).squeeze(-1).squeeze(-1).unsqueeze(1)

                v2 = extractor1.forward_1(v2)
                v2 = reformer2(v2)
                v2 = extractor1.forward_2(v2)
                v2 = nn.functional.adaptive_avg_pool2d(v2, 1).squeeze(-1).squeeze(-1).unsqueeze(1)

                v3 = extractor1(v3)
                v3 = nn.functional.adaptive_avg_pool2d(v3, 1).squeeze(-1).squeeze(-1).unsqueeze(1)

                v4 = extractor1.forward_1(v4)
                v4 = reformer2(v4)
                v4 = extractor1.forward_2(v4)
                v4 = nn.functional.adaptive_avg_pool2d(v4, 1).squeeze(-1).squeeze(-1).unsqueeze(1)

                v5 = extractor1.forward_1(v5)
                v5 = reformer1(v5)
                v5 = extractor1.forward_2(v5)
                v5 = nn.functional.adaptive_avg_pool2d(v5, 1).squeeze(-1).squeeze(-1).unsqueeze(1)

                feat = torch.cat([v1, v2, v3, v4, v5], dim=1)
                features = torch.cat((features, feat), 0)

                fcount += 1
                # Stop after 300 frames in ODV, 130 frames in JVQD, 500 frames in SVQD
                if fcount > video_length:
                    break
            ffmpeg.close()
            features=features.to('cpu').numpy()
            np.save(os.path.join(feature_path,vname[:-4]+'.npy'),features)
