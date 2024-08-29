import torch

from tqdm import tqdm

from module.msa import MSARowAttentionWithPairBias, MSAColumnAttention
from module.util import DropoutRowwise, DropoutColumnwise
from module.transition import MSATransition, PairTransition
from module.outer_product_mean import OuterProductMean
from module.triangle_attention import TriangleAttentionStartingNode, TriangleAttentionEndingNode
from module.triangle_multiplication import TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming

class EvoformerBlock(torch.nn.Module):
    def __init__(self, c_m=64, c_z=128, c_h_m=32, c_h_o=32, c_h_p=32, n_h_m=8, n_h_p=4, t_n=4, m_d=0.15, p_d=0.25, inf=1e9, eps=1e-10):
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_h_o = c_h_o
        self.c_h_m = c_h_m
        self.c_h_p = c_h_p
        self.n_h_m = n_h_m
        self.n_h_p = n_h_p
        self.t_n = t_n
        self.m_d = m_d
        self.p_d = p_d

        self.dropout_row_msa = DropoutRowwise(m_d)
        self.dropout_row_pair = DropoutRowwise(p_d)
        self.dropout_col = DropoutColumnwise(p_d)

        self.msa_row_attn = MSARowAttentionWithPairBias(c_m, c_z, c_h_m, n_h_m)
        self.msa_col_attn = MSAColumnAttention(c_m, c_h_m, n_h_m)
        self.msa_transition = MSATransition(c_m, t_n)

        self.outer_product_mean = OuterProductMean(256, c_z, 32)

        # symmetry를 반영하기 위해서 triangular multiplicative update는 outgoing edge에 대해서 한 번, incoming edge에 대해서 한 번, 총 두 번 수행하게 된다.
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z)
        self.tri_attn_start = TriangleAttentionStartingNode(c_z, c_h_p, n_h_p)
        self.tri_attn_end = TriangleAttentionEndingNode(c_z, c_h_p, n_h_p)
        self.pair_transition = PairTransition(c_z, t_n)

    def forward(self, m, z, msa_mask, pair_mask):
        """
        Algorithm 6: Evoformer stack (Block)

        m: (B, s, i, c)
        z: (B, i, j, c)

        return: [
            m: (B, s, i, c),
            z: (B, i, j, c)
        ]
        """
        m_ = self.dropout_row_msa(self.msa_row_attn(m, z, msa_mask))
        m = m + m_
        m_ = self.msa_col_attn(m, msa_mask)
        m = m + m_
        m_ = self.msa_transition(m, msa_mask)
        m = m + m_
        z_ = self.outer_product_mean(m, msa_mask)
        z = z + z_

        z_ = self.dropout_row_pair(self.tri_mul_out(z, pair_mask))
        z = z + z_
        z_ = self.dropout_row_pair(self.tri_mul_in(z, pair_mask))
        z = z + z_
        z_ = self.dropout_row_pair(self.tri_attn_start(z, pair_mask))
        z = z + z_
        z_ = self.dropout_col(self.tri_attn_end(z, pair_mask))
        z = z + z_
        z_ = self.pair_transition(z, pair_mask)
        z = z + z_

        return m, z


class Evoformer(torch.nn.Module):
    def __init__(self, n_block=48, c_s=384, c_m=256, c_z=128, c_h_m=32, c_h_o=128, c_h_p=32, n_h_m=8, n_h_p=4, t_n=4, m_d=0.15, p_d=0.25, inf=1e9, eps=1e-10):
        super().__init__()

        self.n_block = n_block
        self.c_m = c_m
        self.c_z = c_z
        self.c_h_o = c_h_o
        self.c_h_m = c_h_m
        self.c_h_p = c_h_p
        self.n_h_m = n_h_m
        self.n_h_p = n_h_p
        self.t_n = t_n
        self.m_d = m_d
        self.p_d = p_d

        self.blocks = torch.nn.ModuleList([
                EvoformerBlock(c_m=c_m, c_z=c_z, c_h_m=c_h_m, c_h_o=c_h_o, c_h_p=c_h_p, n_h_m=n_h_m, n_h_p=n_h_p, t_n=t_n, m_d=m_d, p_d=p_d, inf=inf, eps=eps)
                for _ in range(n_block)
        ])

        self.proj_o = torch.nn.Linear(c_m, c_s)
    
    def forward(self, m, z, msa_mask, pair_mask):
        """
        return: [
            m: (B, s, i, c),
            z: (B, i, j, c),
            s: (B, i, c)
        ]
        """
        for block in tqdm(self.blocks):
           m, z = block(m, z, msa_mask, pair_mask)
        
        s = self.proj_o(m[..., 0, :, :])

        return m, z, s

