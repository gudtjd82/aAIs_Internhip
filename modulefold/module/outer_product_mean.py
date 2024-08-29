import torch

class OuterProductMean(torch.nn.Module):
    def __init__(self, c_m=256, c_z=128, c_h=32, eps=1e-3):
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_h = c_h
        self.eps = eps

        self.layer_norm = torch.nn.LayerNorm(c_m)
        self.linear_a = torch.nn.Linear(c_m, c_h)
        self.linear_b = torch.nn.Linear(c_m, c_h)
        self.linear_out = torch.nn.Linear(c_h * c_h, c_z)

    def forward(self, m, msa_mask=None):
        '''
        m: (B, s, r, c_m)

        return (B, r, r, c_z)
        '''
        if msa_mask is None:
            msa_mask = m.new_ones(m.shape[:-1])
        msa_mask = msa_mask.unsqueeze(-1)   # (B, s, r, 1)
        
        m = self.layer_norm(m) # [B, s, r, cm]

         # [B, r, s]

        #   (B, s, r, c_h)   * (B, s, r, c_m)   -> (B, s, r, c_m)
        a = self.linear_a(m) * msa_mask # [B, s, r, cm]
        b = self.linear_b(m) * msa_mask # [B, s, r, cm]

        a = a.transpose(-2, -3)     # (B, r, s, c_h)
        b = b.transpose(-2, -3)     # (B, r, s, c_h)

        # a, b의 outer product
        outer = torch.einsum("...bac,...dae->...bdce", a, b)    # (B, r, r, c_h, c_h)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))     # (B, r, r, c_h * c_h)

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)      # (B, r, r, c_z)

        # msa_mask: (B, s, r, 1)
        norm = torch.einsum("...abc,...adc->...bdc", msa_mask, msa_mask)    # (B, r, r, 1)
        # ...r,s,1, ...r,s,1 -> s,s,1
        norm = norm + self.eps
        outer = outer / norm        # (B, r, r, c_z)

        return outer #* 0

