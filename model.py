import torch
import torch.nn as nn
import torch.nn.functional as F
from se3cnn.blocks import GatedBlock
from lib.utils import rot_to_wignerD
from lib.sample_rotations import sample_rotations_60
R_bin_ctrs = torch.tensor(sample_rotations_60("matrix")).float()

def bin_delta_to_Rs_batch(bin_Rs, R_bin_ctrs):
    """
    args:
    bin_Rs: [B, bins],
    R_bin_ctrs: [B, 3, 3],
    returns:
    rotation matrix: [B, 3, 3],
    """
    def bin_to_R(bin_R, R_bin_ctrs):
        return R_bin_ctrs[torch.argmax(bin_R)]
    Rs = []
    for i in range(len(bin_Rs)):
        Rs.append(bin_to_R(bin_Rs[i], R_bin_ctrs))
    return torch.stack(Rs)

def rotate_field(x, rot, feature):
    """
    rot: rep in R3, R: rep in field
    x: [B, C, D, H, W]
    rot: [B, 3, 3]
    feature: [m1, m2, m3, m4]
    """
    B, C, D, H, W = x.size()
    x = x.permute(0, 1, 4, 3, 2)# (B, C, D, H, W) -> (B, C, W, H, D)
    theta = torch.zeros((B, 3, 4), dtype=torch.float32).cuda()
    theta[:, 0:3, 0:3] = rot.permute(0, 2, 1).contiguous()
    grid = F.affine_grid(theta, x.size())
    x_trans = F.grid_sample(x, grid)
    x_trans = x_trans.permute(0, 1, 4, 3, 2)# (B, C, W, H, D) -> (B, C, D, H, W)

    A = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]).unsqueeze(0).repeat(B, 1, 1).cuda()
    rot = torch.bmm(torch.bmm(A, rot), A.permute(0, 2, 1))
    D0, D1, D2, D3 = rot_to_wignerD(rot, feature)# [(B, 1, 1), (B, 3, 3), (B, 5, 5), (B, 7, 7)]
    Dmat = torch.zeros(B, C, C).float().cuda()

    n_irreps = feature[0]
    for i in range(n_irreps):
        Dmat[:, 1*i:1*(i+1), 1*i:1*(i+1)] = D0
        Dmat[:, 1*n_irreps+3*i:1*n_irreps+3*(i+1), 1*n_irreps+3*i:1*n_irreps+3*(i+1)] = D1
        Dmat[:, 4*n_irreps+5*i:4*n_irreps+5*(i+1), 4*n_irreps+5*i:4*n_irreps+5*(i+1)] = D2
        Dmat[:, 9*n_irreps+7*i:9*n_irreps+7*(i+1), 9*n_irreps+7*i:9*n_irreps+7*(i+1)] = D3
    x_trans = torch.bmm(Dmat.detach(), x_trans.reshape(B, C, -1)).reshape(B, C, D, H, W).contiguous()

    return x_trans


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

def deg_metric(pred_rot, gt_rot):
    pred_rot = quat2mat(pred_rot)# (B, 4) -> (B, 3, 3)
    gt_rot = quat2mat(gt_rot)# (B, 4) -> (B, 3, 3)
    R = torch.bmm(pred_rot, gt_rot.transpose(2, 1))# (B, 3, 3)
    cos_theta = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) - 1) / 2# B
    ang_dis = torch.acos(torch.clamp(cos_theta, -1.0, 1.0)) * 180 / 3.14159# B
    return ang_dis

class AvgSpacial(torch.nn.Module):
    def forward(self, inp):
        # inp [batch, features, x, y, z]
        return inp.reshape(inp.size(0), inp.size(1), -1).mean(-1).contiguous()  # [batch, features]

class Model(torch.nn.Module):
    def __init__(self, num_class, num_k=20):
        super().__init__()
        self.num_k = num_k

        layer_params= {
            'normalization': 'batch',
            'activation': (F.relu, torch.sigmoid),
            'smooth_stride': False}
        self.conv1  = GatedBlock((1, ), (2, 2, 2, 2), size=3, padding=1, stride=1, **layer_params)

        self.conv21 = GatedBlock((2, 2, 2, 2), (4, 4, 4, 4), size=3, padding=1, stride=1, **layer_params)
        self.conv22 = GatedBlock((4, 4, 4, 4), (4, 4, 4, 4), size=3, padding=1, stride=1, **layer_params)
        self.conv23 = GatedBlock((4, 4, 4, 4), (4, 4, 4, 4), size=3, padding=1, stride=1, **layer_params)
        self.conv24 = GatedBlock((4, 4, 4, 4), (4, 4, 4, 4), size=3, padding=1, stride=1, **layer_params)

        self.conv31 = GatedBlock((4, 4, 4, 4), (8, 8, 8, 8), size=3, padding=1, stride=1, **layer_params)
        self.conv32 = GatedBlock((8, 8, 8, 8), (8, 8, 8, 8), size=3, padding=1, stride=1, **layer_params)

        self.conv41 = GatedBlock((8, 8, 8, 8), (16, 16, 16, 16), size=3, padding=1, stride=1, **layer_params)
        self.conv42 = GatedBlock((16, 16, 16, 16), (16, 16, 16, 16), size=3, padding=1, stride=1, **layer_params)

        self.convp1  = GatedBlock((28, 28, 28, 28), (16, 16, 16, 16), size=3, padding=1, stride=1, **layer_params)

        self.conv51  = GatedBlock((16, 16, 16, 16), (32, 32, 32, 32), size=3, padding=1, stride=1, **layer_params)
        self.conv52  = GatedBlock((32, 32, 32, 32), (64, 64, 64, 64), size=3, padding=1, stride=1, **layer_params)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 60)

        self.conv0_c = GatedBlock((16, 16, 16, 16), (16, 16, 16, 16), size=3, padding=1, stride=1, **layer_params)
        self.conv1_c = GatedBlock((16, 16, 16, 16), (32, 32, 32, 32), size=3, padding=1, stride=1, **layer_params)
        self.conv2_c = GatedBlock((32, 32, 32, 32), (128, 128), size=3, padding=1, stride=1, **layer_params)
        self.fc1_c = nn.Linear(512, 256)
        self.bn1_c = nn.BatchNorm1d(256)
        self.fc2_c = nn.Linear(256, 128)
        self.bn2_c = nn.BatchNorm1d(128)
        self.fc3_c = nn.Linear(128, num_class)

        self.avgpool = nn.AvgPool3d(kernel_size=3, padding=1, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.gblpool = AvgSpacial()
        self.relu = nn.ReLU(True)

    def forward(self, vol, gt_Rmat=None, mode='test'):
        feat1 = self.conv1(vol)# B,16,64^3
        feat1_p = self.avgpool(feat1)# B,16,32^3

        feat21 = self.conv21(feat1_p)# B,64,32^3
        feat22 = self.conv22(feat21)# B,64,32^3
        feat22_p = self.avgpool(feat22)# B,64,16^3
        feat23 = self.conv23(feat22_p)# B,64,16^3
        feat24 = self.conv24(feat23)# B,64,16^3

        feat31 = self.conv31(feat24)# B,128,16^3
        feat32 = self.conv32(feat31)# B,128,16^3
        feat3_p = self.avgpool(feat32)# B,128,8^3

        feat41 = self.conv41(feat3_p)# B,256,8^3
        feat42 = self.conv42(feat41)# B,256,8^3

        pyr1, pyr2, pyr3 = feat24, feat32, self.up(feat42)
        type_1 = torch.cat([pyr1[:, 0:4, ...], pyr2[:, 0:8, ...], pyr3[:, 0:16, ...]], dim=1)
        type_2 = torch.cat([pyr1[:, 4:16, ...], pyr2[:, 8:32, ...], pyr3[:, 16:64, ...]], dim=1)
        type_3 = torch.cat([pyr1[:, 16:36, ...], pyr2[:, 32:72, ...], pyr3[:, 64:144, ...]], dim=1)
        type_4 = torch.cat([pyr1[:, 36:64, ...], pyr2[:, 72:128, ...], pyr3[:, 144:256, ...]], dim=1)
        feat_pyr = torch.cat([type_1, type_2, type_3, type_4], dim=1)# B,(28,28,28,28),16^3
        feat_pyr1 = self.convp1(feat_pyr)# B,(16,16,16,16),16^3

        feat_pyr_p = self.avgpool(feat_pyr1)# B,(16,16,16,16),8^3
        feat51 = self.conv51(feat_pyr_p)# B,(32,32,32,32),8^3
        feat52 = self.conv52(feat51)# B,(64,64,64,64),8^3
        feat = self.gblpool(feat52)# B,1024
        feat = self.relu(self.bn1(self.fc1(feat)))
        feat = self.relu(self.bn2(self.fc2(feat)))
        pred_rot_bin = self.fc3(feat)# B,40

        '--------------------------------------------------------------------------'

        if mode == 'train':
            inv_Rmat = gt_Rmat.permute(0, 2, 1).contiguous()
            can_feat = rotate_field(feat_pyr1, inv_Rmat, (16, 16, 16, 16))# B,(16,16,16,16),16^3

            feat_c = self.conv0_c(can_feat)# B,256,16^3
            feat_c = self.avgpool(feat_c)# B,256,8^3
            feat_c = self.conv1_c(feat_c)# B,512,8^3
            feat_c = self.conv2_c(feat_c)# B,512,8^3
            feat_c = self.gblpool(feat_c)# B,512
            feat_c = self.relu(self.bn1_c(self.fc1_c(feat_c)))
            feat_c = self.relu(self.bn2_c(self.fc2_c(feat_c)))
            cls = self.fc3_c(feat_c)# B,40
            return cls, pred_rot_bin

        elif mode == 'test':
            pred_rot_bin_k = torch.topk(pred_rot_bin, k=self.num_k, dim=1)[1]# (B, 60) -> (B, k)
            log_list, cls_list = [], []
            for i in range(self.num_k):
                # (bins, 3, 3) -> (B, 3, 3)
                pred_Rmat = torch.gather(R_bin_ctrs.cuda(), 0, pred_rot_bin_k[:, i][:, None, None].repeat(1, 3, 3))
                inv_Rmat = pred_Rmat.permute(0, 2, 1).contiguous()
                can_feat = rotate_field(feat_pyr1, inv_Rmat, (16, 16, 16, 16))

                feat_c = self.conv0_c(can_feat)# B,256,16^3
                feat_c = self.avgpool(feat_c)# B,256,8^3
                feat_c = self.conv1_c(feat_c)# B,512,8^3
                feat_c = self.conv2_c(feat_c)# B,512,8^3
                feat_c = self.gblpool(feat_c)# B,512
                feat_c = self.relu(self.bn1_c(self.fc1_c(feat_c)))
                feat_c = self.relu(self.bn2_c(self.fc2_c(feat_c)))
                cls = self.fc3_c(feat_c)# B,40

                pred_log, pred_cls = torch.max(cls, dim=1)
                log_list.append(pred_log[:, None])
                cls_list.append(pred_cls[:, None])
            cand_log = torch.cat(log_list, dim=1)# (B, k)
            cand_cls = torch.cat(cls_list, dim=1)# (B, k)
            return cand_log, cand_cls, pred_rot_bin





