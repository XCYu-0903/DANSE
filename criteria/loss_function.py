from joblib import Parallel, delayed
import numpy as np
from pesq import pesq
import torch


def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def phase_losses(phase_r, phase_g):
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))
    return ip_loss, gd_loss, iaf_loss


def eval_pesq(clean_utt, esti_utt, sr):
    try:
        return pesq(sr, clean_utt, esti_utt)
    except Exception:
        return -1


def pesq_score(utts_r, utts_g, h):
    scores = Parallel(n_jobs=30)(
        delayed(eval_pesq)(
            utts_r[i].squeeze().cpu().numpy(),
            utts_g[i].squeeze().cpu().numpy(),
            h.sampling_rate,
        )
        for i in range(len(utts_r))
    )
    return torch.tensor(float(np.mean(scores)))
