import torch


def collate_fn_mhealth(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)          # [B,L,C]
    y = torch.cat(ys, dim=0).long()     # [B]
    padding_mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool)
    return x, y, padding_mask
