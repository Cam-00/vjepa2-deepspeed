# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path


def build_action_block_causal_attention_mask(T, H, W, add_tokens=1):
    """
        Builds a block causal attention mask
        Parameters:
            T: Number of timesteps
            H: Height dimension size
            W: Width dimension size
            add_tokens: Number of additional tokens (default is 1)

        Returns:
            mask: (N, N), N = T * (H*W + add_tokens)
    """

    # Calculate tokens per timestep = additional tokens + spatial tokens (H*W)
    N_T = add_tokens + (H * W)

    # Calculate total tokens = timesteps * tokens per timestep
    N = T * N_T

    # Initialize a zero attention mask matrix (N×N) of boolean type
    mask = torch.zeros(N, N).bool()

    # Create a block mask (N_T×N_T), all 1s mean all positions within the block can attend to each other
    mask_block = torch.ones(N_T, N_T).bool()

    # Set local time window size; T means considering all past timesteps
    local_window_time = T

    # Iterate through all timesteps t1 (current timestep)
    for t1 in range(T):
        # Iterate through allowable timesteps t2 for t1 to attend to (past timesteps)
        # Range is from max(0, t1-local_window_time+1) to t1 (inclusive)
        for t2 in range(max(0, t1 - local_window_time + 1), t1 + 1):
            # Set the corresponding block area in the mask to 1 (allow attention)
            # Row range: t1*N_T to (t1+1)*N_T
            # Column range: t2*N_T to (t2+1)*N_T
            mask[t1 * N_T: (t1 + 1) * N_T, t2 * N_T: (t2 + 1) * N_T] = mask_block

    # Return the constructed attention mask
    return mask  # (N, N)


def rotate_queries_or_keys(x, pos):   # Specific 1D-RoPE implementation
    """
        x : can be q, k or v, shape is [B, num_heads, N, D] where D is even and D = head_dim // 3
            N is the number of tokens (patches)

        pos: (N, ) or (B, num_heads, N)

        Returns: [B, num_heads, N, D]
    """
    B, num_heads, N, D = x.size()  #
    assert D % 2 == 0, "Embedding dimension must be a multiple of 2 for block matrix rotation"

    # -- compute angle for each position
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0  # normalization, element value range [0, 1]
    omega = 1.0 / 10000**omega  # (D/2,) corresponds to θ in RoPE paper
    freq = torch.einsum("..., f -> ... f", pos, omega)  # (N, D/2) or (B, num_heads, N, D/2), outer product

    # -- build rotation matrix and apply
    emb_sin = freq.sin()  # (N, D/2) or (B, num_heads, N, D/2)
    emb_cos = freq.cos()  # (N, D/2) or (B, num_heads, N, D/2)

    emb_sin = emb_sin.squeeze(-1).repeat(1, 1, 1, 2)  # (1, 1, N, D) or (B, num_heads, N, D)
    emb_cos = emb_cos.squeeze(-1).repeat(1, 1, 1, 2)  # (1, 1, N, D) or (B, num_heads, N, D)

    # --
    y = x.unflatten(-1, (-1, 2))  # shape: [B, num_heads, N, D/2, 2], first col is odd index x, second col is even index x
    y1, y2 = y.unbind(dim=-1)     # shape: [B, num_heads, N, D/2], y1 corresponds to odd, y2 corresponds to even
    y = torch.stack((-y2, y1), dim=-1)  # shape: [B, num_heads, N, D/2, 2]
    y = y.flatten(-2)                           # shape: [B, num_heads, N, D]
    return (x * emb_cos) + (y * emb_sin)  # [B, num_heads, N, D] Corresponds to formula (34) in RoPE paper


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
       DropPath is typically used in the main path of residual blocks.
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # DropPath is a regularization technique similar to Dropout, but acts on multi-branch structures in deep learning
        # DropPath randomly "disables" certain branches during forward propagation to enhance generalization
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()   # Activation layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """
        SwiGLU(x) = (SiLU(xW₁ + b₁)) ⊙ (xW₂ + b₂)
        Provides stronger non-linear expressive power than standard FFN (GELU + single linear layer)

        When wide_silu=True:
            Hidden layer dimension is set to 2/3 of standard FFN (maintaining similar parameter count)
            Aligned to multiples of 8 (e.g., input 1024 -> calculate 682 -> adjust to 688)

        Dimension alignment (align_as=8) ensures memory access efficiency
        Dual-branch parallel computation (fc1 and fc2 can execute in parallel)
    """
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.0, wide_silu=True
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


class ACRoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,  # Parameter to control the use of PyTorch's optimized Scaled Dot-Product Attention (SDPA)
        is_causal=False,  # Whether to use causal attention
        grid_size=16,   # patch size
    ):
        super().__init__()
        # Basic parameter settings
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Projection layer definitions
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # QKV joint projection
        self.attn_drop = nn.Dropout(attn_drop)             # Attention dropout
        self.proj = nn.Linear(dim, dim)                    # Output projection
        self.proj_drop_prob = proj_drop                    # Dropout probability for projection layer
        self.proj_drop = nn.Dropout(proj_drop)             # Output dropout
        self.use_sdpa = use_sdpa

        # 3D-RoPE Parameters
        self.d_dim = int(2 * ((head_dim // 3) // 2))  # Ensure even (required for complex pairs in rotation encoding)
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))

        self.grid_size = grid_size    # Base patch size for position normalization
        self.is_causal = is_causal

    def _get_frame_pos(self, ids, H_patches, W_patches):
        """Calculates which frame (time) position each token belongs to"""
        tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches, W_patches):
        """Calculates the height position of each token within a frame"""
        # Remove frame component from ids
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        # --
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches, W_patches):
        """Separates 3D position information into a (time, height, width) triplet"""
        tokens_per_frame = int(H_patches * W_patches)
        tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        # --
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(self, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        """
            x: Input tokens (B, N, C)
                B: Batch size (number of input video clips)
                N: Tokens after embedding, masking, and adding action/state (if mask is not None)
                    N <= T*H*W or (N == T*H*W + 2 or 3), where (T, H, W) is the shape of blocks
                C: Dimension per token
            mask: Visible token position indices [B, M], M < N, generally None here
            attn_mask: Custom attention mask
            T: Time dimension length (number of frames)
            H: Number of patches in height direction
            W: Number of patches in width direction
            action_tokens: Number of action/state tokens (used in video tasks), value is 2 or 3
        """

        B, N, C = x.size()

        # -- compute position of each frame token
        if mask is not None:  # (B, N)
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)  # Expand to multiple heads (B, num_heads, N)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)  # (B, num_heads, N)
        else:  # No mask, all visible
            mask = torch.arange(int(T * H * W), device=x.device)  # (N, )
            d_mask, h_mask, w_mask = self.separate_positions(mask, H, W)  # (N, )

        # -- snap spatial positions to grid size
        # Visible token positions normalized to patch_size baseline
        h_mask *= self.grid_size / H   # self.grid_size is patch size
        w_mask *= self.grid_size / W

        # -- split out action tokens from sequence
        # Action token processing (video tasks)
        if action_tokens > 0:
            x = x.view(B, -1, action_tokens + H * W, C)  # (B, T, action_tokens+H*W, C), action_tokens = 2 or 3
            action_q, action_k, action_v = [], [], []
            for i in range(action_tokens):
                """Process each action/state token individually"""
                a = x[:, :, i : i + 1, :].flatten(1, 2)           # (B, T, C)
                # Note action tokens do not work with masking
                # -- compute qkv for action tokens and rotate
                # (B, T, C) -> (B, T, 3*C) -> (B, T, 3, num_heads, head_dim) -> (3, B, num_heads, T, head_dim), C = num_heads * head_dim
                qkv = self.qkv(a).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]    # (B, num_heads, T, head_dim)
                # -- Action tokens: Rotary Positional Encoding applied only to time dimension
                # (Strongly correlated with time, weak spatial correlation; spatial RoPE might cause overfitting)
                qd = rotate_queries_or_keys(q[..., : self.d_dim], pos=torch.arange(T, device=x.device))  # (B, num_heads, T, d_dim)
                kd = rotate_queries_or_keys(k[..., : self.d_dim], pos=torch.arange(T, device=x.device))
                qr = q[..., self.d_dim :]  # (B, num_heads, T, head_dim - d_dim)
                kr = k[..., self.d_dim :]
                # Elements of action_q/k/v are tensor (B, num_heads, T, 1, head_dim)
                action_q += [torch.cat([qd, qr], dim=-1).view(B, self.num_heads, T, 1, -1)]
                action_k += [torch.cat([kd, kr], dim=-1).view(B, self.num_heads, T, 1, -1)]
                action_v += [v.view(B, self.num_heads, T, 1, -1)]

            action_q = torch.cat(action_q, dim=3).flatten(2, 3)  # (B, num_heads, n*T, head_dim), n = 2 or 3
            action_k = torch.cat(action_k, dim=3).flatten(2, 3)
            action_v = torch.cat(action_v, dim=3).flatten(2, 3)

            # Exclude action/state tokens
            x = x[:, :, action_tokens:, :].flatten(1, 2)  # (B, T*H*W, C)

        # -- compute qkv for frame tokens and rotate
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (B, num_heads, T*H*W, head_dim)

        """Implement 3D-RoPE for frame tokens"""
        s = 0
        # Rotate depth
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask)  # (B, num_heads, T*H*W, d_dim)
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        # Rotate height dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        # Rotate width dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)  # (B, num_heads, T*H*W, head_dim)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)      # (B, num_heads, T*H*W, head_dim)
            k = torch.cat([kd, kh, kw], dim=-1)

        if action_tokens > 0:

            def merge_(tx, ta):
                """tx, tx in [B, num_heads, N, D]"""
                tx = tx.view(B, self.num_heads, T, H * W, -1)  # [B, num_heads, T, H*W, D]
                ta = ta.view(B, self.num_heads, T, action_tokens, -1)  # [B, num_heads, T, action_tokens, D]
                return torch.cat([ta, tx], dim=3).flatten(2, 3)  # [B, num_heads, T*(action_tokens + H*W), D]

            q = merge_(q, action_q)  # (B, num_heads, N, head_dim), N = T(n+H*W), n = 2 or 3
            k = merge_(k, action_k)
            v = merge_(v, action_v)

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                # Scaled dot product attention using optimized CUDA kernels
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,   # Whether to scale the qk dot product result
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,   # Parameter to control optimized self-attention calculation via SDPA
        grid_size=14,    # Patch grid size: the grid after input image is patched
        is_causal=False,  # Whether to use causal attention
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # Learns q, k, v simultaneously, so output feature dim is 3x input
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        # --
        self.d_dim = int(2 * ((head_dim // 3) // 2))  # Ensure even for 3D-RoPE
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame   # ids.shape

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        # Remove frame component from ids
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        # --
        return ids // tokens_per_row  # ids.shape

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        # --
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids   # ids.shape

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        # B: Batch size (number of input video clips)
        # N: Tokens (patches or grid) after embedding and masking (if mask is not None)
        #    N <= T*H*W, where (T, H, W) is block shape
        # C: Dimension per token
        B, N, C = x.size()
        grid_depth = int(N // (self.grid_size * self.grid_size))
        # Shape after self.qkv(): [B, N, 3*C]
        # Shape after unflatten: [B, N, 3, num_heads, 3*C // (3 * num_heads)] last dim is head feature dimension
        # Shape after permute: [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D] where D = head_dim = C // num_heads

        if mask is not None:  # (B, N)
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)  # (B, num_heads, N)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)  # (B, num_heads, N)
        else:  # (N, )
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(int(grid_depth * self.grid_size * self.grid_size), device=x.device)  # (N, )
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)      # (N, )
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)  # (N, )

        s = 0

        ## Concrete 3D-RoPE implementation
        # Rotate depth
        qd = rotate_queries_or_keys(q[..., s : s + self.d_dim], pos=d_mask)  # [B, num_heads, N, d_dim], d_dim = head_dim / 3
        kd = rotate_queries_or_keys(k[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        # Rotate height dim
        qh = rotate_queries_or_keys(q[..., s : s + self.h_dim], pos=h_mask)
        kh = rotate_queries_or_keys(k[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        # Rotate width dim
        qw = rotate_queries_or_keys(q[..., s : s + self.w_dim], pos=w_mask)
        kw = rotate_queries_or_keys(k[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim

        # Combine rotated dimension
        if s < self.head_dim:  # Handle cases where head_dim is not a multiple of 6 (3 dims * 2 for complex pairs)
            qr = q[..., s:]  # qr not rotated
            kr = k[..., s:]  # kr not rotated
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)  # [B, num_heads, N, head_dim]
            k = torch.cat([kd, kh, kw], dim=-1)  # [B, num_heads, N, head_dim]

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
            attn = attn.softmax(dim=-1)                    # [B, num_heads, N, N]
            attn = self.attn_drop(attn)   # [B, num_heads, N, N] step not in original transformer
            x = attn @ v                  # [B, num_heads, N, D]

        x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, C] where C = num_heads * D
        x = self.proj(x)       # [B, N, C]
        x = self.proj_drop(x)  # [B, N, C]
        return x               # [B, N, C]


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x, mask=None, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ACBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = ACRoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # FFN layer
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, attn_mask=None, T=None, H=None, W=None, action_tokens=0):
        y = self.norm1(x)
        if isinstance(self.attn, ACRoPEAttention):
            y = self.attn(y, mask=mask, attn_mask=attn_mask, T=T, H=H, W=W, action_tokens=action_tokens)
        else:
            y = self.attn(y, mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)
        y = self.norm2(x)
        x = x + self.drop_path(self.mlp(y))
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_rope:
            self.attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        if isinstance(self.attn, RoPEAttention):
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
        else:
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask)
        x = x + self.drop_path(y)  # drop_path + residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # LayerNorm + FFN + drop_path + residual connection
        return x    # [B, N, D]


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        # self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            q = xattn @ v

        q = q.transpose(1, 2).reshape(B, n, C)
        return q


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q


class CrossAttentionBlockPatched(nn.Module):
    """
        Cross Attention Block: With Attention Temperature adjustment / Residual Scaling / Cosine Attention
            - Pre-LN: LayerNorm applied separately to q/kv
            - Cosine Attention (Optional): Q/K are unit-vectorized; scores are no longer divided by sqrt(d_k)
            - Attention Temperature tau: Softens softmax
            - Residual Scaling alpha: ReZero style, stabilizes gradients
            - Includes MLP branch (likewise with Pre-LN + residual scaling)
        """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.0, norm_layer=nn.LayerNorm,
                 use_cosine_qk: bool = True, tau: float = 2.0, qkv_bias: bool = False,
                 init_alpha_attn: float = 1e-2, init_alpha_mlp: float = 1e-2, act_layer=nn.GELU,
                 eps: float = 1e-6):
        super().__init__()

        # Parameter validation
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        assert tau > 0, f"tau must be positive, got {tau}"
        assert init_alpha_attn > 0 and init_alpha_mlp > 0, "alpha init values must be positive"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_cosine_qk = use_cosine_qk  # Cosine attention
        self.tau = tau                      # Attention temperature
        self.eps = eps

        # Normalization layers
        self.q_ln = norm_layer(dim, eps=eps)
        self.kv_ln = norm_layer(dim, eps=eps)
        self.mlp_ln = norm_layer(dim, eps=eps)

        # Attention projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        # Residual scaling parameters
        self.alpha_attn = nn.Parameter(torch.ones(1) * init_alpha_attn)
        self.alpha_mlp = nn.Parameter(torch.ones(1) * init_alpha_mlp)

        # MLP branch
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        self.dropout = nn.Dropout(drop)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Projection layers using Xavier initialization
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # MLP layer initialization
        nn.init.xavier_uniform_(self.mlp.fc1.weight)
        nn.init.xavier_uniform_(self.mlp.fc2.weight)
        nn.init.constant_(self.mlp.fc1.bias, 0)
        nn.init.constant_(self.mlp.fc2.bias, 0)

    def _shape_heads(self, x):
        B, L, D = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        return x

    def forward(self, q_tokens, kv_tokens, return_attn_stats: bool = False):
        # Pre-LN
        q_norm = self.q_ln(q_tokens)
        kv_norm = self.kv_ln(kv_tokens)

        # Projections
        q = self.q_proj(q_norm)
        k = self.k_proj(kv_norm)
        v = self.v_proj(kv_norm)

        # Reshape for multi-head attention
        q = self._shape_heads(q)
        k = self._shape_heads(k)
        v = self._shape_heads(v)

        # Compute attention scores
        if self.use_cosine_qk:
            q = F.normalize(q, p=2, dim=-1, eps=self.eps)
            k = F.normalize(k, p=2, dim=-1, eps=self.eps)
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.tau
        else:
            scale = (self.head_dim ** 0.5) * self.tau
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Attention weights and context
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        # Combine heads and project
        context = context.transpose(1, 2).contiguous()
        context = context.view(q_tokens.size(0), q_tokens.size(1), self.dim)
        context = self.out_proj(context)
        context = self.dropout(context)

        # Residual connection with scaling
        output = q_tokens + self.alpha_attn * context

        # MLP branch
        mlp_output = self.mlp(self.mlp_ln(output))
        output = output + self.alpha_mlp * mlp_output

        if return_attn_stats:
            stats = self._compute_attention_stats(scores, attn_weights)
            for key, value in stats.items():
                print(f"{key}: {value}")