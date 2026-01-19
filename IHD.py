import torch
import math
from util import torch_dct

def DCTBlur(x, t):
    eps = 1e-4
    a = 0.5

    orig_dtype = x.dtype
    device = x.device
    t = t.clamp_min(eps)
    x = x.to(torch.float32)
    t = t.to(dtype=torch.float32)
    dtype = x.dtype

    if x.dim() == 4:
        B, C, H, W = x.shape
        x_in = x
        has_channel = True
    elif x.dim() == 3:
        B, H, W = x.shape
        x_in = x.unsqueeze(1)
        B, C, H, W = x_in.shape
        has_channel = False

    i = torch.arange(H, device=device, dtype=dtype).view(H, 1)
    j = torch.arange(W, device=device, dtype=dtype).view(1, W)
    lamb = - (math.pi ** 2) * (i**2 / (H**2) + j**2 / (W**2))

    lamb_view = lamb.view(1, 1, H, W)
    t_view = t.view(B, 1, 1, 1)

    dct_x = torch_dct.dct_2d(x_in, norm='ortho') 
    C_x_t = t_view ** (-a * lamb_view) 

    A = C_x_t * dct_x                   
    B_spec = lamb_view * A              

    stacked = torch.cat([A, B_spec], dim=1)   
    spatial = torch_dct.idct_2d(stacked, norm='ortho')  

    u_t = spatial[:, :C, :, :]         
    delta_ut = spatial[:, C:, :, :]      

    if not has_channel:
        u_t = u_t.squeeze(1)            
        delta_ut = delta_ut.squeeze(1)        

    return u_t.to(orig_dtype), delta_ut.to(orig_dtype)
