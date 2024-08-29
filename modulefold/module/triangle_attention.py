import torch

class TriangleAttentionStartingNode(torch.nn.Module):
    def __init__(self, c=32, c_h=16, n_head=4):
        super().__init__()
        self.c = c
        self.c_h = c_h
        self.n_head = n_head

        self.layer_norm = torch.nn.LayerNorm(c)

        self.query = torch.nn.Linear(c, c_h * n_head, bias=False)
        self.key = torch.nn.Linear(c, c_h * n_head, bias=False)
        self.value = torch.nn.Linear(c, c_h * n_head, bias=False)

        self.bias = torch.nn.Linear(c, n_head, bias=False)
        self.gate = torch.nn.Linear(c, c_h * n_head)
        self.output = torch.nn.Linear(c_h * n_head, c)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, z, pair_mask=None):
        """
        Algorithm 13: Triangular gated self-attention around starting node

        z: (B, i, j, c)

        return: (B, i, j, c)
        """

        z = self.layer_norm(z) # (B, i, j, c)

        q, k, v = self.query(z), self.key(z), self.value(z) # (B, i, j, c * h)
        b = self.bias(z) # (B, i, j, h)
        # b.shape = 1, 256, 256, 4
        g = self.sigmoid(self.gate(z)) # (B, i, j, c * h)

        # B, i, j, _ = q.shape
        q_shape = q.shape[:-1]
        h, c, c_h = self.n_head, self.c, self.c_h

        q = q.view(*q_shape, h, c_h) # (B, i, j, h, c)
        k = k.view(*q_shape, h, c_h)
        v = v.view(*q_shape, h, c_h)
        g = g.view(*q_shape, h, c_h)
        b = b.view(*q_shape[:-2], 1, *q_shape[-2:], h)#.transpose(-2,-4).transpose(-2,-3)       # (B, 1, i, j , h)

        if q.dim() == 4:
            a = torch.einsum("i q h c, i v h c -> i q v h", q, k) / (c_h**0.5) # (B, i, r_q, r_v, h)
            if pair_mask != None:
                a = a +  (1e9 * (pair_mask - 1))[..., :, None, :, None]
            a = a + b
        else:
            a = torch.einsum("b i q h c, b i v h c -> b i q v h", q, k) / (c_h**0.5)
            if pair_mask != None:
                a = a + (1e9 * (pair_mask - 1))[..., :, None, :, None]
            a = a + b
        a = torch.nn.functional.softmax(a, dim=-2) # (B, i, r_q, r_v, h)
        
        if a.dim() == 4:
            o = g * torch.einsum("i q v h, i v h c -> i q h c", a, v) # (B, i, r_q, h, c)
        else:
            o = g * torch.einsum("b i q v h, b i v h c -> b i q h c", a, v)
        o = o.view(*q_shape, h * c_h) # (B, i, j, h * c)
        o = self.output(o)
        
        return o  # (B, i, j, c_z)
    

class TriangleAttentionEndingNode(torch.nn.Module):
    def __init__(self, c=32, c_h=16, n_head=4):
        super().__init__()
        self.c = c
        self.c_h = c_h
        self.n_head = n_head

        self.layer_norm = torch.nn.LayerNorm(c)

        self.query = torch.nn.Linear(c, c_h * n_head, bias=False)
        self.key = torch.nn.Linear(c, c_h * n_head, bias=False)
        self.value = torch.nn.Linear(c, c_h * n_head, bias=False)

        self.bias = torch.nn.Linear(c, n_head, bias=False)
        self.gate = torch.nn.Linear(c, c_h * n_head)
        self.output = torch.nn.Linear(c_h * n_head, c)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, z, pair_mask=None):
        """
        Algorithm 14: Triangular gated self-attention around ending node

        z: (B, i, j, c)

        return: (B, i, j, c)
        """

        z = z.transpose(-2, -3) # Ending node, flips i and j
        z = self.layer_norm(z) # (B, i, j, c)

        q, k, v = self.query(z), self.key(z), self.value(z) # (B, i, j, c * h)
        b = self.bias(z) # (B, i, j, h)
        g = self.sigmoid(self.gate(z)) # (B, i, j, c * h)

        q_shape = q.shape[:-1]
        h, c, c_h = self.n_head, self.c, self.c_h

        q = q.view(*q_shape, h, c_h) # (B, i, j, h, c)
        k = k.view(*q_shape, h, c_h)
        v = v.view(*q_shape, h, c_h)
        g = g.view(*q_shape, h, c_h)
        b = b.view(*q_shape[:-2], 1, *q_shape[-2:], h)

        if q.dim() == 4:
            a = torch.einsum("i q h c, i v h c -> i q v h", q, k) / (c_h**0.5) # (B, i, r_q, r_v, h)
            if pair_mask != None:
                a = a +  (1e9 * (pair_mask - 1))[..., :, None, :, None]
            a = a + b
        else:
            a = torch.einsum("b i q h c, b i v h c -> b i q v h", q, k) / (c_h**0.5)
            if pair_mask != None:
                a = a + (1e9 * (pair_mask - 1))[..., :, None, :, None]
            a = a + b
        a = torch.nn.functional.softmax(a, dim=-2) # (B, i, r_q, r_v, h)

        if a.dim() == 4:
            o = g * torch.einsum("i q v h, i v h c -> i q h c", a, v) # (B, i, r_q, h, c)
        else:
            o = g * torch.einsum("b i q v h, b i v h c -> b i q h c", a, v)
        o = o.view(*q_shape, h * c_h) # (B, i, j, h * c)
        # with open('minifold.pickle', 'wb') as f:
        #     pickle.dump(o, f, pickle.HIGHEST_PROTOCOL)
        o = self.output(o).transpose(-2, -3)
        
        return o  # (B, i, j, c_z)