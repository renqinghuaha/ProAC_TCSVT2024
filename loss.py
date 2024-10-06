import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelContrastLoss(nn.Module):
    def __init__(self, cfg):
        super(PixelContrastLoss, self).__init__()
        self.cfg = cfg

    def Samples_generation(self, st_feat, st_label):
        src_mask = F.interpolate(
                input=st_label.unsqueeze(1).float(),
                size=st_feat.shape[2:],
                mode='nearest')
        src_mask = src_mask.contiguous().view(-1).long()
        assert not src_mask.requires_grad
        n, c, h, w = st_feat.size()
        src_feat_NxC = st_feat.permute(0, 2, 3, 1).contiguous().view(-1, c)
        feat_sim = []
        label_sim = []
        for lb in range(19):
            src_idxs = (src_mask == lb).nonzero().squeeze(-1)
            if len(src_idxs) > 0:
                this_feat = src_feat_NxC[src_idxs].clone()
                this_label = src_mask[src_idxs].clone()
                feat_sim.append(this_feat)
                label_sim.append(this_label)
        feat_sim = torch.cat(feat_sim, dim=0)
        label_sim = torch.cat(label_sim, dim=0)
        return feat_sim, label_sim

    def contrast_cos_calc(self, x1, x2):
        X_ = F.normalize(x1, p=2, dim=-1)
        Y_ = F.normalize(x2, p=2, dim=-1)
        out = torch.matmul(X_, Y_.T)
        return out

    def process_label(self, label):
        n = label.size()[0]
        pred1 = torch.zeros(n, 19).cuda()
        pred1 = pred1.scatter_(1, label.unsqueeze(1).long(), 1)
        return pred1

    # proto-anchor
    def PixACLoss(self, feats, labels, queue):
        n, c = feats.size()
        mask = self.process_label(labels)
        _queue = queue.contiguous().view(-1, c)
        anchor_dot_contrast = self.contrast_cos_calc(feats, _queue)
        anchor_dot_contrast = torch.div(anchor_dot_contrast, 0.1)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        neg_mask = 1.0 - mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)
        exp_logits = torch.exp(logits)
        log_prob = torch.log(exp_logits + neg_logits) - logits
        loss = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        loss = loss.mean()
        return loss

    # proto-anchor
    def ProACLoss(self, feats, labels, queue):
        n, c = feats.size()
        mask = self.process_label(labels)
        mask = mask.permute(1, 0)
        _queue = queue.contiguous().view(-1, c)
        contrast = self.contrast_cos_calc(feats, _queue)
        anchor_dot_contrast = contrast.permute(1, 0)
        anchor_dot_contrast = torch.div(anchor_dot_contrast, 0.1)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        neg_mask = 1.0 - mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)
        exp_logits = torch.exp(logits)
        log_prob = torch.log(exp_logits + neg_logits) - logits
        loss = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        this_label_ids = torch.unique(labels)
        loss = loss.sum() / len(this_label_ids)
        return loss

    def forward(self, feats, labels, queue):
        src_feat_sim, src_label_sim = self.Samples_generation(feats, labels)
        loss_contrast = self.ProACLoss(src_feat_sim, src_label_sim, queue)
        return loss_contrast