import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import Model
from dataloader import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os

from lib.sample_rotations import sample_rotations_60
R_bin_ctrs = torch.tensor(sample_rotations_60("matrix")).float().cuda()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = '')
parser.add_argument('--data_path', type=str, default = '')
parser.add_argument('--batch_size', type=int, default = 32)
parser.add_argument('--workers', type=int, default = 8)
parser.add_argument('--learning_rate', default = 0.005)
parser.add_argument('--lr_decay', default = [1, 0.1, 0.01])
parser.add_argument('--lr_steps', default = [10, 15])
parser.add_argument('--nepoch', type=int, default = 20)
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

def get_learning_rate(epoch):
    assert len(lr_decay) == len(lr_steps) + 1
    for lim, lr in zip(lr_steps, lr_decay):
        if epoch < lim:
            return lr * learning_rate
    return lr_decay[-1] * learning_rate

if __name__ == '__main__':

    if opt.dataset == 'pm40':
        num_class = 40
    elif opt.dataset == 'ps15':
        num_class = 15
    learning_rate = opt.learning_rate
    lr_decay = opt.lr_decay
    lr_steps = opt.lr_steps

    torch.backends.cudnn.benchmark = True
    writer = SummaryWriter(logdir='./runs')

    TRAIN_DATASET = DataLoader(dataset=opt.dataset, root=opt.data_path, split='train')
    TEST_DATASET = DataLoader(dataset=opt.dataset, root=opt.data_path, split='test')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)

    classifier = nn.DataParallel(Model(num_class=num_class).cuda())
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    print('# classifier parameters:', sum(param.numel() for param in classifier.parameters()))

    global_step = 0
    for epoch in range(0, opt.nepoch):

        total_bin = 0
        total_correct = 0
        total_seen = 0
        lr = get_learning_rate(epoch)
        for p in optimizer.param_groups:
            p['lr'] = lr
        classifier = classifier.train()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            vol, gt_cls, gt_rot, gt_noi = data
            vol, gt_cls, gt_rot, gt_noi = \
                vol.cuda(), gt_cls.cuda().long(), gt_rot.cuda(), gt_noi.cuda()
            optimizer.zero_grad()
            gt_rot_bin = Rs_to_bin_delta_batch(quat2mat(gt_rot), R_bin_ctrs, knn=True)# (B, 4) -> B
            gt_noi_bin = Rs_to_bin_delta_batch(quat2mat(gt_noi), R_bin_ctrs)# (B, 4) -> B
            gt_Rmat_noi = torch.gather(R_bin_ctrs, 0, gt_noi_bin[:, None, None].repeat(1, 3, 3))# (bins, 3, 3) -> (B, 3, 3)

            pred_cls, pred_rot_bin = classifier(vol, gt_Rmat=gt_Rmat_noi, mode='train')
            cls_loss = F.cross_entropy(pred_cls, gt_cls)
            rot_bin_loss = -(gt_rot_bin * F.log_softmax(pred_rot_bin, -1)).sum(dim=1).mean()  

            total_loss = cls_loss + rot_bin_loss   
            total_loss.backward()
            optimizer.step()

            global_step += 1
            total_correct += torch.sum(torch.argmax(pred_cls, 1) == gt_cls).item()
            total_bin += torch.sum(torch.argmax(pred_rot_bin, 1) == gt_rot_bin.argmax(1)).item()
            total_seen += pred_cls.shape[0]

            if global_step%100==0:
                writer.add_scalar('train/cls_loss', cls_loss.item(), global_step)
                writer.add_scalar('train/rot_loss', rot_bin_loss.item(), global_step)

        train_ins_acc = total_correct/float(total_seen)
        train_bin_acc = total_bin/float(total_seen)
        print('Epoch: %d, Train Ins Acc: %f' % (epoch+1, train_ins_acc))
        print('Epoch: %d, Train Bin Acc: %f' % (epoch+1, train_bin_acc))
        writer.add_scalar('train/ins_acc', train_ins_acc, epoch+1)
        writer.add_scalar('train/bin_acc', train_bin_acc, epoch+1)

        ###################################################################################################
        with torch.no_grad():
            classifier = classifier.eval()

            total_correct = 0
            total_bin = 0            
            total_seen = 0   
            total_correct_class = [0 for _ in range(num_class)]    
            total_seen_class = [0 for _ in range(num_class)]  
            for batch_id, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
                vol, gt_cls, gt_rot, gt_noi = data
                vol, gt_cls, gt_rot, gt_noi = \
                    vol.cuda(), gt_cls.cuda().long(), gt_rot.cuda(), gt_noi.cuda()
                gt_rot_bin = Rs_to_bin_delta_batch(quat2mat(gt_rot), R_bin_ctrs)# (B, 4) -> B
                
                cand_log, cand_cls, pred_rot_bin = classifier(vol)
                final_cls = torch.gather(cand_cls, 1, torch.argmax(cand_log, 1)[:, None]).view(-1)

                total_correct += torch.sum(final_cls == gt_cls).item()
                total_bin += torch.sum(torch.argmax(pred_rot_bin, 1) == gt_rot_bin).item()
                total_seen += final_cls.shape[0]
                for i in range(final_cls.shape[0]):
                    l = gt_cls[i]
                    total_correct_class[l] += (final_cls[i] == l).item()
                    total_seen_class[l] += 1

            test_ins_acc = total_correct / float(total_seen)
            test_cls_acc = np.mean(np.array(total_correct_class)/np.array(total_seen_class, dtype=np.float))
            test_bin_acc = total_bin / float(total_seen)
            print('Epoch: %d, Test Ins Acc: %f' % (epoch+1, test_ins_acc))
            print('Epoch: %d, Test Cls Acc: %f' % (epoch+1, test_cls_acc))
            print('Epoch: %d, Test Bin Acc: %f' % (epoch+1, test_bin_acc))
            writer.add_scalar('test/ins_acc', test_ins_acc, epoch+1)
            writer.add_scalar('test/cls_acc', test_cls_acc, epoch+1)  
            writer.add_scalar('test/bin_acc', test_bin_acc, epoch+1)  

            if not os.path.exists(f'./ckpt/{opt.dataset}'):
                os.makedirs(f'./ckpt/{opt.dataset}')
            torch.save(classifier.state_dict(), './ckpt/{0}/model_{1}_{2}.pth'.format(opt.dataset, epoch+1, test_ins_acc))
         
           