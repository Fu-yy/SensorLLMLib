import torch


def collate_fn_mhealth(batch):
    xs, ys,mean,var = zip(*batch)
    x = torch.stack(xs, dim=0)          # [B,L,C]
    y = torch.cat(ys, dim=0).long()     # [B]

    mean = torch.stack(mean, dim=0)
    var = torch.stack(var, dim=0)

    padding_mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool)
    return x, y, padding_mask,mean,var
