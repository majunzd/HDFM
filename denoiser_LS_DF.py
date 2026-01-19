import torch
import torch.nn as nn
torch.set_float32_matmul_precision('high')
from model import models
from IHD import DCTBlur


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            layersync_layers=args.layersync_layers, 
            layersync_lambda=args.layersync_lambda,
            return_layersync=args.return_layersync,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)
    
    def forward(self, x, labels):
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale
        x_t_blur, delta_t_blur = DCTBlur(x, t)
        z = t * x_t_blur + (1 - t) * e
        s = (1 - t).clamp_min(self.t_eps)

        v = (x_t_blur - z) / s  - delta_t_blur

        x_pred, ls_loss  = self.net(z, t.flatten(), labels_dropped)
        x_pred_t_blur, delta_pred_t_blur = DCTBlur(x_pred, t)
        v_pred = (x_pred_t_blur - z) / s - delta_pred_t_blur

        vloss = (v - v_pred) ** 2
        vloss = vloss.mean(dim=(1, 2, 3)).mean()

        loss = vloss + ls_loss
        return loss

    @torch.no_grad()
    def generate(self, labels):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z


    @torch.no_grad()
    def _forward_sample(self, z, t, labels):
        s = (1.0 - t).clamp_min(self.t_eps)
        x_cond, _ = self.net(z, t.flatten(), labels)
        x_cond_t_blur, delta_cond_t_blur = DCTBlur(x_cond, t)
        v_base_cond = (x_cond_t_blur - z) / s

        uncond_labels = torch.full_like(labels, self.num_classes)
        x_uncond, _ = self.net(z, t.flatten(), uncond_labels)
        x_uncond_t_blur, delta_uncond_t_blur = DCTBlur(x_uncond, t)
        v_base_uncond = (x_uncond_t_blur - z) / s

        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        base = v_base_uncond + cfg_scale_interval * (v_base_cond - v_base_uncond)
        delta = delta_uncond_t_blur + cfg_scale_interval * (delta_cond_t_blur - delta_uncond_t_blur)

        return base, delta

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels):
        dt = (t_next - t)
        base_t, delta_t = self._forward_sample(z, t, labels)
        v_pred0 = base_t - 0.5 * delta_t
        z_euler = z + dt * v_pred0
        base_n, delta_n = self._forward_sample(z_euler, t_next, labels)
        v_full_t = base_t - delta_t
        v_full_n = base_n - delta_n
        v_no_t   = base_t
        v_no_n   = base_n
        r_full = v_full_n - v_full_t
        r_no   = v_no_n   - v_no_t
        eps = 1e-12
        sig_full = r_full.pow(2).mean(dim=(1,2,3), keepdim=True) + eps
        sig_no   = r_no.pow(2).mean(dim=(1,2,3), keepdim=True) + eps
        beta = (sig_no / (sig_full + sig_no)).clamp(0.05, 0.95)
        v = base_t - beta * delta_t
        return z + dt * v


    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels):
        dt = (t_next - t)

        base_t, delta_t = self._forward_sample(z, t, labels)
        v_pred0 = base_t - 0.5 * delta_t
        z_euler = z + dt * v_pred0

        base_n, delta_n = self._forward_sample(z_euler, t_next, labels)
        
        v_full_t = base_t - delta_t
        v_full_n = base_n - delta_n

        v_no_t   = base_t
        v_no_n   = base_n

        r_full = v_full_n - v_full_t
        r_no   = v_no_n   - v_no_t

        eps = 1e-12
        sig_full = r_full.pow(2).mean(dim=(1,2,3), keepdim=True) + eps
        sig_no   = r_no.pow(2).mean(dim=(1,2,3), keepdim=True) + eps

        beta = sig_no / (sig_full + sig_no)
        beta = beta.clamp(0.05, 0.95)

        base_heun  = 0.5 * (base_t  + base_n)
        delta_heun = 0.5 * (delta_t + delta_n)

        v = base_heun - beta * delta_heun
        return z + dt * v

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
