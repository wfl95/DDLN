from torch.nn import Module, Parameter
import math
import torch
import torch.nn as nn


# DDLN
class NoiseFilter(nn.Module):
    def __init__(self, log=None, milestone0=11, gap_epoch=2, top_class=10, ema_t=0.01, alpha=0.02):
        super().__init__()
        self.log = log
        self.milestone0 = int(milestone0)       # first learning rate decay
        self.gap_epoch = int(gap_epoch)
        self.top_class = int(top_class)   # k
        self.t = float(ema_t)             # lambda
        self.alpha = float(alpha)         # slack alpha

        self.register_buffer("minNoise", torch.zeros(1))       # T'_min
        self.register_buffer("maxNoise", torch.full((1,), -1.0))  # T'_max
        self.register_buffer("noiseThresh", torch.zeros(1))    # T_n

    @torch.no_grad()
    def _ema_update_(self, buf, val):
        # buf <- (1-t)*buf + t*val
        buf.mul_(1.0 - self.t).add_(val, alpha=self.t)

    def _log_scalar(self, name, val):
        if self.log is not None:
            self.log(name, val, on_step=True, on_epoch=True, logger=True)

    @torch.no_grad()
    def _update_noise_bounds(self, gt, cosine, label, current_epoch):
        # retained clean set Q_n
        cleanIdx = gt.squeeze(1) > self.maxNoise.item()
        if not cleanIdx.any():
            self._log_scalar("minNoise", self.minNoise)
            self._log_scalar("maxNoise", self.maxNoise)
            self._log_scalar("noiseThresh", self.noiseThresh)
            return

        # cleaned sub mini batch
        cos_sel = cosine[cleanIdx].clone()  # [Bc, C]
        lbl_sel = label[cleanIdx].view(-1, 1)  # [Bc, 1]

        # ---- estimate T'_min ----
        if current_epoch == 0:
            row_min = cos_sel.min(dim=1).values  # [Bc]
            self._ema_update_(self.minNoise, row_min.mean().view(1))

        # non-target cosine similarities
        cos_sel.scatter_(1, lbl_sel, float("-inf"))

        # ---- estimate T'_max ----
        k = min(self.top_class, cos_sel.size(1) - 1)
        if k > 0:
            topk_vals = cos_sel.topk(k, dim=1, largest=True).values  # [Bc, k]
            kth_vals = topk_vals[:, -1]  # [Bc]
            maxNoise = torch.maximum(self.maxNoise, kth_vals.mean().view(1) + self.alpha)
            self._ema_update_(self.maxNoise, maxNoise)

        self._log_scalar("minNoise", self.minNoise)
        self._log_scalar("maxNoise", self.maxNoise)
        self._log_scalar("noiseThresh", self.noiseThresh)

    def forward(self, cosine, label, current_epoch):
        # cosine: [B, C], label: [B]
        device = cosine.device
        B = label.size(0)

        gt = cosine[torch.arange(B, device=device), label].view(-1, 1)  # [B,1]

        # update local T'_min and T'_max from current batch
        self._update_noise_bounds(cosine, label, current_epoch)

        # progressive interpolation: T'_n = T'_min + beta * (T'_max - T'_min)
        n_max = max(self.milestone0 - self.gap_epoch, 1)
        beta = min(current_epoch, n_max) / float(n_max)
        thresh = self.minNoise + (self.maxNoise - self.minNoise) * beta

        # EMA smoothing: T_n = (1-lambda)T_{n-1} + lambda T'_n
        self._ema_update_(self.noiseThresh, thresh)

        # use smoothed threshold for filtering
        effective = self.noiseThresh if current_epoch > 0 else gt.new_tensor([-1.0])
        mask = (gt.squeeze(1) >= effective.item())

        self._log_scalar("thresh_local", thresh)
        self._log_scalar("thresh", self.noiseThresh)

        return cosine[mask], label[mask]

class CosFace(nn.Module):

    def __init__(self, s=64., m=0.35, denoiseEnable=False):
        super(CosFace, self).__init__()

        self.m = m  
        self.s = s  
        self.eps = 1e-4

        self.denoiseEnable = denoiseEnable
        if self.denoiseEnable:
            self.noise_filter = NoiseFilter(milestone0=11, gap_epoch=2)

        print('init CosFace with ')
        print('self.m', self.m)
        print('self.s', self.s)

    def forward(self, cosine, label, current_epoch):

        if self.denoiseEnable: # DDLN
            cosine, label= self.noise_filter(cosine, label, current_epoch=current_epoch)

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        cosine = cosine - m_hot
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m, label


class ArcFace(Module):

    def __init__(self, s=64., m=0.5, denoiseEnable=False):
        super(ArcFace, self).__init__()
        self.m = m  
        self.s = s  
        self.eps = 1e-4

        self.denoiseEnable = denoiseEnable
        if self.denoiseEnable:
            self.noise_filter = NoiseFilter(milestone0=11, gap_epoch=3)


    def forward(self, cosine, label, current_epoch):
        if self.denoiseEnable:   # DDLN
            cosine, label= self.noise_filter(cosine, label, current_epoch=current_epoch)

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        theta = cosine.acos()

        theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi-self.eps)
        cosine_m = theta_m.cos()
        scaled_cosine_m = cosine_m * self.s

        return scaled_cosine_m, label



#### Head
def build_head(head_type, m, s):
    denoiseEnable = 'denoise' in head_type.lower()
    if 'arcface' in head_type.lower():
        head = ArcFace(m=m,s=s, denoiseEnable=denoiseEnable)
    elif 'cosface' in head_type.lower():
        head = CosFace(m=m,s=s, denoiseEnable=denoiseEnable)
    else:
        raise ValueError('not a correct head type', head_type)
    return head

