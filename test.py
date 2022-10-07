import argparse
import torch
import torch.nn as nn
import numpy as np
from model import Model
from dataloader import DataLoader
from tqdm import tqdm

from lib.sample_rotations import sample_rotations_60
R_bin_ctrs = torch.tensor(sample_rotations_60("matrix")).float().cuda()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = '')
parser.add_argument('--data_path', type=str, default = '')
parser.add_argument('--model', type=str, default = '')
parser.add_argument('--num_k', type=int, default = 20)
opt = parser.parse_args()

def Rs_to_bin_delta_batch(Rs, R_bin_ctrs, knn=False):
    def R_to_bin_delta(R=None, R_bin_ctrs=None, theta1=0.4, theta2=0.2, knn=False):
        def geodesic_dists(R_bin_ctrs, R):
            internal = 0.5 * (torch.diagonal(torch.matmul(R_bin_ctrs, torch.transpose(R, -1, -2)), 
                dim1=-1, dim2=-2).sum(-1) - 1.0)
            internal = torch.clamp(internal, -1.0, 1.0)
            return torch.acos(internal)
        dists = geodesic_dists(R_bin_ctrs, R)
        if knn:
            bin_R = torch.zeros(R_bin_ctrs.shape[0]).cuda()
            delta_R = torch.zeros(R_bin_ctrs.shape).cuda()
            _, nn4 = torch.topk(dists, k=4, largest=False)
            bin_R[nn4] = theta2
            bin_R[nn4[0]] = theta1
        else:
            bin_R = torch.argmin(dists)
            delta_R = R[..., :3, : 3].matmul(R_bin_ctrs[bin_R].t())
        return bin_R, delta_R

    bin_Rs = []
    for i in range(len(Rs)):
        bin_R, _ = R_to_bin_delta(Rs[i], R_bin_ctrs, knn=knn)
        bin_Rs.append(bin_R)
    return torch.stack(bin_Rs)

def quat2mat(q):
    B = q.size(0)
    R = torch.cat(((1.0 - 2.0*(q[:, 2]**2 + q[:, 3]**2)).view(B, 1), \
            (2.0*q[:, 1]*q[:, 2] - 2.0*q[:, 0]*q[:, 3]).view(B, 1), \
            (2.0*q[:, 0]*q[:, 2] + 2.0*q[:, 1]*q[:, 3]).view(B, 1), \
            (2.0*q[:, 1]*q[:, 2] + 2.0*q[:, 3]*q[:, 0]).view(B, 1), \
            (1.0 - 2.0*(q[:, 1]**2 + q[:, 3]**2)).view(B, 1), \
            (-2.0*q[:, 0]*q[:, 1] + 2.0*q[:, 2]*q[:, 3]).view(B, 1), \
            (-2.0*q[:, 0]*q[:, 2] + 2.0*q[:, 1]*q[:, 3]).view(B, 1), \
            (2.0*q[:, 0]*q[:, 1] + 2.0*q[:, 2]*q[:, 3]).view(B, 1), \
            (1.0 - 2.0*(q[:, 1]**2 + q[:, 2]**2)).view(B, 1)), dim=1).view(B, 3, 3)
    return R 

def pc2vol(points, vsize=64, radius=1.0):
    B = points.shape[0]# B,1024,3
    voxel = 2 * radius/ float(vsize)
    locations = ((points + radius) / voxel).type(torch.int64)# B,1024,3
    locations = locations[:, :, 0]*vsize*vsize+locations[:, :, 1]*vsize+locations[:, :, 2]# B,1024
    vol = torch.zeros((B, vsize**3)).cuda()
    vol = vol.scatter(1, locations, torch.ones(locations.shape).cuda())
    vol = vol.view(B, 1, vsize, vsize, vsize).contiguous()
    return vol

if __name__ == '__main__':

    if opt.dataset == 'pm40':
        num_class = 40
    elif opt.dataset == 'ps15':
        num_class = 15

    torch.backends.cudnn.benchmark = True
    TEST_DATASET = DataLoader(dataset=opt.dataset, root=opt.data_path, split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    classifier = nn.DataParallel(Model(num_class=num_class, num_k=opt.num_k).cuda())
    classifier.load_state_dict(torch.load(opt.model), strict=False)
    classifier = classifier.eval()
    print('# classifier parameters:', sum(param.numel() for param in classifier.parameters()))

    with torch.no_grad():
        total_correct = np.zeros(opt.num_k)
        total_bin = np.zeros(opt.num_k)         
        total_seen = 0
        for batch_id, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
            vol, gt_cls, gt_rot, gt_noi = data
            vol, gt_cls, gt_rot, gt_noi = \
                vol.cuda(), gt_cls.cuda().long(), gt_rot.cuda(), gt_noi.cuda()
            gt_rot_bin = Rs_to_bin_delta_batch(quat2mat(gt_rot), R_bin_ctrs)# (B, 4) -> B

            cand_log, cand_cls, pred_rot_bin = classifier(vol)# (B, k), (B, k)
            for i in range(opt.num_k):
                final_cls = torch.gather(cand_cls[:, 0:i+1], 1, torch.argmax(cand_log[:, 0:i+1], 1)[:, None]).view(-1) # (B, )
                total_correct[i] += torch.sum(final_cls == gt_cls).item()

                pred_rot_bin_k = torch.topk(pred_rot_bin, k=i+1, dim=1)[1]# (B, 60) -> (B, i+1)
                total_bin[i] += (pred_rot_bin_k == gt_rot_bin[:, None]).any(1).sum()
            total_seen += final_cls.shape[0]

        test_ins_acc = total_correct / float(total_seen)
        test_bin_acc = total_bin / float(total_seen)
        for i in range(opt.num_k):
            print('k=%d, Test Ins Acc: %f, Test Bin Top-k Acc: %f' % (i+1, test_ins_acc[i], test_bin_acc[i]))
         
           