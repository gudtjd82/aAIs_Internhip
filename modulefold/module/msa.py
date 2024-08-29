import torch

# MSA representation의 row 단위로 gated self-attention을 수행하는데, 
# pair representation(z)의 linear projection을 bias로 반영하여 attention weight를 구하게 된다. 
# 정보 전달: Pair representation(=bias) -> MSA representation
class MSARowAttentionWithPairBias(torch.nn.Module):
    def __init__(self, c_m=256, c_z=128, c_h=4, n_head=8):
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_h = c_h
        self.n_head = n_head

        self.layer_norm = torch.nn.LayerNorm(c_m)
        self.layer_norm_b = torch.nn.LayerNorm(c_z)
        self.proj_q = torch.nn.Linear(c_m, c_h * n_head, bias=False)    # Query
        self.proj_k = torch.nn.Linear(c_m, c_h * n_head, bias=False)    # Key
        self.proj_v = torch.nn.Linear(c_m, c_h * n_head, bias=False)    # Value
        self.proj_b = torch.nn.Linear(c_z, n_head, bias=False)          # Pair Bias
        self.proj_g = torch.nn.Linear(c_m, c_h * n_head)                # Gate
        self.proj_o = torch.nn.Linear(c_h * n_head, c_m)                # Output

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, m, z, msa_mask=None):
        """
        Algorithm 7: MSA row-wise gated self-attention with pair bias

        m: (B, s, i, c)
        z: (B, i, j, c)
        msa_mask: (B, s, i)

        return: (B, s, i, c)
        """
        m = self.layer_norm(m)
        

        q, k, v = self.proj_q(m), self.proj_k(m), self.proj_v(m)

        b = self.proj_b(self.layer_norm_b(z)) # (B, i, j, h)
        gate = self.sigmoid(self.proj_g(m))

        # B, s, i, _ = q.shape
        q_shape = q.shape[:-1]      # [B, s, i]
        h, c_h = self.n_head, self.c_h

        q = q.reshape(*q_shape, h, c_h)                             # (B, s, i, h, c)
        k = k.reshape(*q_shape, h, c_h)                             # (B, s, i, h, c)
        v = v.reshape(*q_shape, h, c_h)                             # (B, s, i, h, c)
        b = b.reshape(*q_shape[:-2], 1, q_shape[-1], q_shape[-1], h) # (B, 1, i, j, h)
        gate = gate.reshape(*q_shape, h, c_h)                       # (B, s, i, h, c)

        if q.dim() == 4:
            a = torch.einsum("s i h c, s j h c -> s i j h", q, k) / (c_h**0.5) # (B, s, i, j, h), Q /cdot K (Query와 Key의 내적)
            if msa_mask != None:
                # msa_mask에서 0인 부분 -> 매우 작은 음수 -> a를 masking
                a = a + (1e9 * (msa_mask-1))[..., :, None, :, None] # (B, s, 1, i, 1)
            a = a+ b
        else:
            a = torch.einsum("b s i h c, b s j h c -> b s i j h", q, k) / (c_h**0.5)
            if msa_mask != None:
                a = a + (1e9 * (msa_mask-1))[..., :, None, :, None]
            a = a + b

        a = torch.nn.functional.softmax(a, dim=-2) # (B, s, i, j, h)

        if a.dim() == 4:
            o = gate * torch.einsum("s i j h, s j h c -> s i h c", a, v) # (B, s, i, h, c), softmax(QK^T) /cdot V
            #           Attention
        else:
            o = gate * torch.einsum("b s i j h, b s j h c -> b s i h c", a, v) # (B, s, i, h, c)
            #           Attention
        o = o.reshape(*q_shape, h * c_h) # (B, s, i, h * c)
        o = self.proj_o(o)

        return o # (B, s, i, c)

# MSA representation의 column 단위로 gated self-attention을 수행한다.
class MSAColumnAttention(torch.nn.Module):
    def __init__(self, c=32, c_h=4, n_head=8):
        super().__init__()

        self.c = c
        self.c_h = c_h
        self.n_head = n_head

        self.layer_norm = torch.nn.LayerNorm(c)

        self.proj_q = torch.nn.Linear(c, c_h* n_head, bias=False)
        self.proj_k = torch.nn.Linear(c, c_h* n_head, bias=False)
        self.proj_v = torch.nn.Linear(c, c_h* n_head, bias=False)
        self.proj_g = torch.nn.Linear(c, c_h * n_head)
        self.proj_o = torch.nn.Linear(c_h * n_head, c)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, m, msa_mask=None):
        """
        Algorithm 8: MSA column-wise gated self-attention

        m: (B, s, i, c)

        return: (B, s, i, c)
        """


        m = self.layer_norm(m)
        if msa_mask == None:
            msa_mask = m.new_ones(m.shape[:-1])
        msa_mask = msa_mask.transpose(-1, -2)   # (B, s, c, i)

        q, k, v = self.proj_q(m), self.proj_k(m), self.proj_v(m)
        gate = self.sigmoid(self.proj_g(m))

        # B, s, i, _ = q.shape
        q_shape = q.shape[:-1]
        h, c, c_h = self.n_head, self.c, self.c_h

        q = q.view(*q_shape, h, c_h)
        k = k.view(*q_shape, h, c_h)
        v = v.view(*q_shape, h, c_h)
        gate = gate.view(*q_shape, h, c_h)

        # 1 128 256 8 32 | 1 256 128 8 32
        # 1 128 256 8 32 | 1 256 128 8 32
        # 1 128 128 256 8 | 1 256 8 128 128
        if q.dim() == 4:
            a = torch.einsum("s i h c, t i h c -> i h s t", q, k) / (c_h**0.5) # (B, s, t, i, h)
            if msa_mask != None:
                a = a + (1e9 * (msa_mask-1)) [..., :, None, None, :]
            # print(a.shape) # 194 8 516 516
        else:
            a = torch.einsum("b s i h c, b t i h c -> b i h s t", q, k) / (c_h**0.5) # (B, s, t, i, h)
            if msa_mask != None:
                a = a + (1e9 * (msa_mask-1)) [..., :, None, None, :]
        a = torch.nn.functional.softmax(a, dim=-1)

        v = v.transpose(-4, -3)
        if a.dim() == 4:
            tmp = torch.einsum("i h d s, i s h c -> i h d c", a, v)
        else:
            tmp = torch.einsum("b i h d s, b i s h c -> b i h d c", a, v)
        tmp = tmp.transpose(-2, -3)
        tmp = tmp.transpose(-4, -3)
        o = gate * tmp # (B, N_s, N_r, h, c)
        o = o.view(*q_shape, h * c_h)
        o = self.proj_o(o)

        return o  # (B, s, i, c)


class MSAColumnGlobalAttention(torch.nn.Module):
    def __init__(self, c=8, c_h=1, n_head=8):
        super().__init__()

        self.c = c
        self.c_h = c_h
        self.n_head = n_head

        self.layer_norm = torch.nn.LayerNorm(c)
        self.proj_q = torch.nn.Linear(c, c_h * n_head, bias=False)
        self.proj_k = torch.nn.Linear(c, c_h, bias=False)
        self.proj_v = torch.nn.Linear(c, c_h, bias=False)
        self.proj_g = torch.nn.Linear(c, c_h * n_head)
        self.proj_o = torch.nn.Linear(c_h * n_head, c)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, m, mask=None):
        """
        Algorithm 19: MSA global column-wise gated self-attention

        m: (B, s, i, c)

        return: (B, s, i, c)
        """

        if mask is None:
            mask = torch.ones(m.shape[:-1], dtype=m.dtype, device=m.device).detach()

        m = m.transpose(-2, -3)
        mask = mask.transpose(-1, -2)

        m = self.layer_norm(m)
        q = torch.sum(m * mask.unsqueeze(-1), dim=-2) / (
            torch.sum(mask, dim=-1)[..., None] + 1e-5
        )

        q, k, v = self.proj_q(q), self.proj_k(m), self.proj_v(m)
        g = self.sigmoid(self.proj_g(m))

        h, c, c_h = self.n_head, self.c, self.c_h
        q = q * (c_h ** (-0.5))

        q_shape = q.shape[:-1]
        q = q.view(*q_shape, h, c_h)
        g = g.view(*g.shape[:-1], h, c_h)

        bias = (1e9 * (mask - 1))[..., :, None, :]

        a = torch.matmul(q, k.transpose(-1, -2))
        a = a + bias
        a = torch.nn.functional.softmax(a, dim=-1)
        o = torch.matmul(a, v)
        o = o.unsqueeze(-3) * g
        o = o.view(*o.shape[:-2], h*c_h)
        o = self.proj_o(o)
    

        o = o.transpose(-2, -3)

        return o # (B, s, i, c)