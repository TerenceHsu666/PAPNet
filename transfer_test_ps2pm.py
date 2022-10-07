import argparse
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
from model import Model
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--ps15_model', type=str, default = '')
parser.add_argument('--pm40_path', type=str, default = '')
parser.add_argument('--num_k', type=int, default = 20)
opt = parser.parse_args()

def pc2vol(points, vsize=64, radius=1.0):
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol

class DataLoader(Dataset):
    def __init__(self, root):
        self.points = np.load(os.path.join(root, 'test_points.npy'))
        self.labels = np.load(os.path.join(root, 'test_labels.npy'))

        keep_idx = np.concatenate([\
            np.where(self.labels==1)[0],\
            np.where(self.labels==2)[0],\
            np.where(self.labels==4)[0],\
            np.where(self.labels==8)[0],\
            np.where(self.labels==12)[0],\
            np.where(self.labels==14)[0],\
            np.where(self.labels==19)[0],\
            np.where(self.labels==22)[0],\
            np.where(self.labels==23)[0],\
            np.where(self.labels==29)[0],\
            np.where(self.labels==30)[0],\
            np.where(self.labels==33)[0],\
            np.where(self.labels==35)[0]], axis=0)
        self.points = self.points[keep_idx]
        self.labels = self.labels[keep_idx]
                                             
        self.labels[np.where(self.labels==1)[0]] = 0 
        self.labels[np.where(self.labels==2)[0]] = 1           
        self.labels[np.where(self.labels==4)[0]] = 9 
        self.labels[np.where(self.labels==8)[0]] = 2
        self.labels[np.where(self.labels==12)[0]] = 3
        self.labels[np.where(self.labels==14)[0]] = 4
        self.labels[np.where(self.labels==19)[0]] = 5
        self.labels[np.where(self.labels==22)[0]] = 6        
        self.labels[np.where(self.labels==23)[0]] = 7
        self.labels[np.where(self.labels==29)[0]] = 10  
        self.labels[np.where(self.labels==30)[0]] = 11 
        self.labels[np.where(self.labels==33)[0]] = 13 
        self.labels[np.where(self.labels==35)[0]] = 14    

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        cls = self.labels[index]

        pts = self.points[index][:, 0:3]
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        radius = np.max(np.linalg.norm(pts, axis=1))
        pts = pts / radius

        vol = pc2vol(pts) # N*3 -> 64*64*64

        return vol[None, :, :, :].astype(np.float32), \
               cls.astype(np.int32)

if __name__ == '__main__':
    TEST_DATASET = DataLoader(root=opt.pm40_path)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    
    classifier = nn.DataParallel(Model(num_class=15).cuda())
    classifier.load_state_dict(torch.load(opt.ps15_model), strict=True)
    classifier = classifier.eval()

    total_correct = np.zeros(opt.num_k)         
    total_seen = 0
    with torch.no_grad():
        for batch_id, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
            vol, gt_cls = data
            vol, gt_cls = vol.cuda(), gt_cls.cuda().long()
            
            cand_log, cand_cls, pred_rot_bin = classifier(vol)# (B, k), (B, k)
            for i in range(opt.num_k):
                final_cls = torch.gather(cand_cls[:, 0:i+1], 1, torch.argmax(cand_log[:, 0:i+1], 1)[:, None]).view(-1) # (B, )
                total_correct[i] += torch.sum(final_cls == gt_cls).item()
            total_seen += final_cls.shape[0]

        test_ins_acc = total_correct / float(total_seen)
        for i in range(opt.num_k):
            print('k=%d, Test Ins Top-k Acc: %f' % (i+1, test_ins_acc[i]))
         