import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from mamba_ssm import selective_scan_fn


class Frequency_Guided_Modulation_Module(nn.Module):

    def __init__(self, d_model, reduction=4):
        super().__init__()
        self.d_model = d_model

        self.freq_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // reduction, d_model, 1),
            nn.Sigmoid()
        )

        self.adaptive_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(d_model, d_model // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // reduction, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, spatial_feat, freq_feat):

        freq_enhanced = freq_feat * self.freq_conv(freq_feat)

        combined_feat = spatial_feat + freq_enhanced
        weights = self.adaptive_weight(combined_feat)
        spatial_weight = weights[:, 0:1, :, :]
        freq_weight = weights[:, 1:2, :, :]
        output = spatial_weight * spatial_feat + freq_weight * freq_enhanced

        return output, freq_weight.squeeze()


class Frequency_Encoding_Module(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.low_freq_weight = nn.Parameter(torch.ones(1))
        self.high_freq_weight = nn.Parameter(torch.ones(1))

        self.low_freq_proj = nn.Linear(d_model, d_model)
        self.high_freq_proj = nn.Linear(d_model, d_model)

        self.freq_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )


        self.weight_norm = nn.Sigmoid()

    def forward(self, x, H, W):
        B, L, D = x.shape


        x_img = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W)


        x_freq = torch.fft.rfft2(x_img, dim=(-2, -1))
        x_freq_mag = torch.abs(x_freq)


        spatial_feat = rearrange(x_img, 'b d h w -> b (h w) d')


        low_weight = self.weight_norm(self.low_freq_weight)
        high_weight = self.weight_norm(self.high_freq_weight)

        freq_h, freq_w = x_freq.shape[-2], x_freq.shape[-1]
        low_freq_mask = torch.zeros_like(x_freq_mag)
        center_h, center_w = freq_h // 4, freq_w // 4
        low_freq_mask[:, :, :center_h, :center_w] = 1

        low_freq = x_freq * low_freq_mask
        low_freq_spatial = torch.fft.irfft2(low_freq, s=(H, W), dim=(-2, -1))
        low_freq_feat = self.low_freq_proj(rearrange(low_freq_spatial, 'b d h w -> b (h w) d'))
        low_freq_feat = low_freq_feat * low_weight

        high_freq_mask = torch.ones_like(x_freq_mag)
        high_freq_mask[:, :, :center_h, :center_w] = 0

        high_freq = x_freq * high_freq_mask
        high_freq_spatial = torch.fft.irfft2(high_freq, s=(H, W), dim=(-2, -1))
        high_freq_feat = self.high_freq_proj(rearrange(high_freq_spatial, 'b d h w -> b (h w) d'))
        high_freq_feat = high_freq_feat * high_weight

        combined_freq = torch.cat([spatial_feat, low_freq_feat, high_freq_feat], dim=-1)
        freq_enhanced = self.freq_fusion(combined_freq)


        return freq_enhanced, x_freq_mag, freq_weights


class MambaVisionMixer(nn.Module):
    """
        MambaVisionMixer
        -----------------
        This module implements the mixing block inspired by the MambaVision framework
        proposed in:

            Ali Hatamizadeh and Jan Kautz,
            "MambaVision: A Hybrid Mamba-Transformer Vision Backbone",
            CVPR 2025.
    """
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
            use_frequency=True,
            safe_mode=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.use_frequency = use_frequency
        self.safe_mode = safe_mode

        # SSM组件
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)

        # 初始化dt参数
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

        if self.use_frequency:
            try:

                self.freq_ratio = nn.Parameter(torch.tensor(0.5))
                self.freq_ratio_norm = nn.Sigmoid()

                self.freq_ssm = Frequency_Encoding_Module(d_model=self.d_inner // 2)
                self.freq_adaptive = Frequency_Guided_Modulation_Module(self.d_inner // 2)
                self.freq_gate = nn.Sequential(
                    nn.Linear(self.d_inner // 2, self.d_inner // 4),
                    nn.ReLU(),
                    nn.Linear(self.d_inner // 4, 1),
                    nn.Sigmoid()
                )
            except Exception as e:
                self.use_frequency = False

    def forward(self, hidden_states, H=None, W=None):
        original_shape = hidden_states.shape

        if hidden_states.dim() == 4:
            B, H, W, D = hidden_states.shape
            hidden_states = rearrange(hidden_states, 'b h w d -> b (h w) d')
        elif hidden_states.dim() == 3:
            if len(original_shape) == 3:
                B, L, D = hidden_states.shape

                if H is not None and W is not None:
                    if L == H * W:

                        pass
                    else:
                        if L == H * W:
                            pass
                        else:
                            raise ValueError(f"L={L} doesn't match H*W={H * W}. "
                                             f"Please check if H, W are window sizes for windowed input.")
                else:
                    H = W = int(math.sqrt(L))
                    if H * W != L:
                        raise ValueError(f"Cannot infer square H, W from L={L}. "
                                         f"Please provide H, W explicitly.")
            else:
                raise ValueError(f"Unexpected input shape: {original_shape}")
        else:
            raise ValueError(f"Input should be 3D (B, L, D) or 4D (B, H, W, D), got {hidden_states.dim()}D")

        seqlen = H * W

        current_B, current_L, current_D = hidden_states.shape
        if current_L != seqlen:
            raise ValueError(f"Sequence length {current_L} doesn't match H*W={seqlen}. "
                             f"Input shape: {original_shape}, H={H}, W={W}")
        B = current_B

        A = -torch.exp(self.A_log.float())

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        x = rearrange(x, "b l d -> b d l")
        z = rearrange(z, "b l d -> b d l")

        try:
            x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias,
                                padding='same', groups=self.d_inner // 2))
            z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias,
                                padding='same', groups=self.d_inner // 2))

            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"🔍 Debug: After conv1d - x.shape={x.shape}, z.shape={z.shape}")

        except Exception as e:
            print(f"❌ Error in convolution: {e}")
            print(f"   x.shape: {x.shape}, z.shape: {z.shape}")
            print(f"   conv1d_x.weight.shape: {self.conv1d_x.weight.shape}")
            print(f"   d_inner: {self.d_inner}, groups: {self.d_inner // 2}")
            raise e

        freq_weights_info = {}
        if self.use_frequency and hasattr(self, 'freq_ssm'):
            try:
                x_seq = rearrange(x, "b d l -> b l d")
                freq_enhanced_x, freq_mag, freq_weights = self.freq_ssm(x_seq, H, W)
                freq_enhanced_x = rearrange(freq_enhanced_x, "b l d -> b d l")

                adaptive_freq_ratio = self.freq_ratio_norm(self.freq_ratio)

                freq_importance = self.freq_gate(rearrange(freq_enhanced_x, "b d l -> b l d")).mean(dim=1)  # (B, 1)

                if x.shape[-1] != H * W or freq_enhanced_x.shape[-1] != H * W:
                    if hasattr(self, '_debug_mode') and self._debug_mode:
                        print(f"⚠️ Warning: Shape mismatch in frequency processing")
                        print(f"  x.shape={x.shape}, freq_enhanced_x.shape={freq_enhanced_x.shape}")
                        print(
                            f"  Expected L=H*W={H * W}, but got x.L={x.shape[-1]}, freq.L={freq_enhanced_x.shape[-1]}")
                    freq_weights_info = {'status': 'shape_mismatch_skipped'}

                else:
                    x_spatial = rearrange(x, "b d l -> b d h w", h=H, w=W)
                    x_freq = rearrange(freq_enhanced_x, "b d l -> b d h w", h=H, w=W)

                    x_fused, freq_adaptive_weight = self.freq_adaptive(x_spatial, x_freq)
                    x = rearrange(x_fused, "b d h w -> b d l")


                    freq_gate_weight = freq_importance.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                    final_freq_weight = adaptive_freq_ratio * freq_gate_weight
                    x = x * (1 + final_freq_weight)

                    freq_weights_info = {
                        'status': 'success',
                        'adaptive_freq_ratio': adaptive_freq_ratio.item(),
                        'low_freq_weight': freq_weights['low_freq_weight'],
                        'high_freq_weight': freq_weights['high_freq_weight'],
                        'avg_freq_importance': freq_importance.mean().item(),
                        'avg_adaptive_weight': freq_adaptive_weight.mean().item()
                    }

            except Exception as e:
                if hasattr(self, '_debug_mode') and self._debug_mode:
                    print(f"⚠️ Warning: Error in frequency processing: {e}")
                    print(f"  Skipping frequency enhancement for this layer")
                freq_weights_info = {'status': 'error', 'error_msg': str(e)}

        try:
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
            B_ssm = rearrange(B_ssm, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C_ssm = rearrange(C_ssm, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"🔍 Debug: SSM params - dt.shape={dt.shape}, B.shape={B_ssm.shape}, C.shape={C_ssm.shape}")

            # 选择性扫描
            y = selective_scan_fn(x,
                                  dt,
                                  A,
                                  B_ssm,
                                  C_ssm,
                                  self.D.float(),
                                  z=None,
                                  delta_bias=self.dt_proj.bias.float(),
                                  delta_softplus=True,
                                  return_last_state=None)

        except Exception as e:
            print(f"❌ Error in SSM processing: {e}")
            print(f"   x.shape: {x.shape}")
            print(f"   seqlen: {seqlen}")
            raise e

        try:
            y = torch.cat([y, z], dim=1)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)

            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"🔍 Debug: Final output shape = {out.shape}")

        except Exception as e:
            print(f"❌ Error in final output processing: {e}")
            print(f"   y.shape: {y.shape}, z.shape: {z.shape}")
            raise e

        if hasattr(self, '_return_weights') and self._return_weights:
            return out, freq_weights_info
        return out



def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x


class GMambaBlock(nn.Module):
    def __init__(self,
                 dim,
                 counter,
                 transformer_blocks,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 window_size=7,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.window_size = window_size

        if counter in transformer_blocks:
            None
        else:
            self.mixer = MambaVisionMixer(d_model=dim,
                                          d_state=8,
                                          d_conv=3,
                                          expand=1,
                                          use_frequency=True,
                                          )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

    def forward(self, x):
        B, C, H, W = x.shape
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W

        x_windowed = window_partition(x, self.window_size)  # (num_windows*B, window_size*window_size, C)
        x_windowed = x_windowed + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x_windowed)))
        x_windowed = x_windowed + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_windowed)))

        x = window_reverse(x_windowed, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()

        return x
