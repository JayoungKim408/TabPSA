import torch
import torch.nn.functional as F
from torch import nn, einsum
# from models.polynomials import *
from einops import rearrange
from models.modules import MLP, exists, PolyTransformer

class TabTransformer_TabPSA(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = nn.SELU(),
        num_special_tokens = 1,
        attn_dropout = 0.,
        ff_dropout = 0.,
        K = 5,
        polynomial = "power",
    ):

        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations
        self.categories = categories
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)
            self.categ_embed = nn.Embedding(total_tokens+1, dim)

        # continuous
        self.num_continuous = num_continuous
        if num_continuous > 0:
            self.num_continuous = num_continuous
            self.norm = nn.LayerNorm(num_continuous)

        # transformer
        self.transformer = PolyTransformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            K = K,
            polynomial = polynomial
        )

        # mlp to logits
        input_size = (dim * self.num_categories) + num_continuous
        l = input_size // 8

        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act = mlp_act)

    def forward(self, x_categ, x_cont, return_attn = False):
        xs = []
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        attns = 0
        if self.num_unique_categories > 0:
            x_categ = torch.where(x_categ==0, 0, x_categ+self.categories_offset)
            x_categ_embed = self.categ_embed(x_categ.long())
            x, attns = self.transformer(x_categ_embed, return_attn = True)
            flat_categ = x.flatten(1)
            xs.append(flat_categ)

        if self.num_continuous > 0:
            normed_cont = self.norm(x_cont)
            xs.append(normed_cont)

        x = torch.cat(xs, dim = -1)

        logits = self.mlp(x)

        if not return_attn:
            return logits

        return logits, attns, x