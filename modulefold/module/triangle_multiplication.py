import torch

class TriangleMultiplicationOutgoing(torch.nn.Module):
    def __init__(self, c=128):
        super().__init__()

        self.c = c

        self.layer_norm = torch.nn.LayerNorm(c)
        self.layer_norm_out = torch.nn.LayerNorm(c)

        self.proj_a = torch.nn.Linear(c, c)
        self.gate_a = torch.nn.Linear(c, c)
        self.proj_b = torch.nn.Linear(c, c)
        self.gate_b = torch.nn.Linear(c, c)
        self.gate = torch.nn.Linear(c, c)
        self.proj_o = torch.nn.Linear(c, c)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, z, pair_mask=None):
        """
        Algorithm 11: Triangular multiplicative update using "outgoing" edges

        z: (B, i, j, c)

        return: (B, i, j, c)
        """

        if pair_mask == None:
            pair_mask = z.new_ones(z.shape[:-1])
        pair_mask = pair_mask.unsqueeze(-1)     # (B< i, j, 1)

        z = self.layer_norm(z)


        a = self.proj_a(z) * self.sigmoid(self.gate_a(z)) * pair_mask
        b = self.proj_b(z) * self.sigmoid(self.gate_b(z)) * pair_mask
        a_std = a.std()
        b_std = b.std()
        if(a_std != 0. and b_std != 0.):
            a = a / a.std()
            b = b / b.std()
        gate = self.sigmoid(self.gate(z))
        o = gate * self.proj_o(self.layer_norm_out(torch.einsum("... i k c, ... j k c -> ... i j c", a, b)))

        return o


class TriangleMultiplicationIncoming(torch.nn.Module):
    def __init__(self, c=128):
        super().__init__()

        self.c = c

        self.layer_norm = torch.nn.LayerNorm(c)
        self.layer_norm_out = torch.nn.LayerNorm(c)

        self.proj_a = torch.nn.Linear(c, c)
        self.gate_a = torch.nn.Linear(c, c)
        self.proj_b = torch.nn.Linear(c, c)
        self.gate_b = torch.nn.Linear(c, c)
        self.gate = torch.nn.Linear(c, c)
        self.proj_o = torch.nn.Linear(c, c)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, z, pair_mask=None):
        """
        Algorithm 12: Triangular multiplicative update using "incoming" edges

        z: (B, i, j, c)

        return: (B, i, j, c)
        """

        if pair_mask == None:
            pair_mask = z.new_ones(z.shape[:-1])
        pair_mask = pair_mask.unsqueeze(-1)

        z = self.layer_norm(z)


        a = self.proj_a(z) * self.sigmoid(self.gate_a(z)) * pair_mask
        b = self.proj_b(z) * self.sigmoid(self.gate_b(z)) * pair_mask
        a_std = a.std()
        b_std = b.std()
        if(a_std != 0. and b_std != 0.):
            a = a / a.std()
            b = b / b.std()
        gate = self.sigmoid(self.gate(z))
        o = gate * self.proj_o(self.layer_norm_out(torch.einsum("... k i c, ... k j c -> ... i j c", a, b)))

        return o