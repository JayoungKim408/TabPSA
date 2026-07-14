
import torch
import torch.nn as nn
from torch import Tensor
# from scipy.linalg import eigvals
# from torch.linalg import eigvals


class PolyConvFrame(nn.Module):
    '''
    A framework for polynomial graph signal filter.
    Args:
        conv_fn: the filter function, like PowerConv, LegendreConv,...
        depth (int): the order of polynomial.
        cached (bool): whether or not to cache the adjacency matrix. 
        alpha (float):  the parameter to initialize polynomial coefficients.
        fixed (bool): whether or not to fix to polynomial coefficients.
    '''
    def __init__(self,
                 conv_fn_type,
                 depth: int = 3,
                 cached: bool = True,
                 alpha: float = 1.0,
                 beta: float = 0.2,
                 fixed: float = True,):
        super().__init__()
        self.depth = depth
        self.basetheta = 1.0

        self.thetas = [nn.Parameter(torch.tensor(beta * (1-beta)**i, requires_grad=True)) for i in range(self.depth)  ] 
        self.thetas.append(nn.Parameter(torch.tensor( beta** self.depth, requires_grad=True)))
        self.thetas = nn.ParameterList(self.thetas)


        self.cached = cached
        self.adj = None
        self.H = []

        if conv_fn_type == 'power':
            self.conv_fn = self.PowerConv
        elif conv_fn_type == "legendre":
            self.conv_fn = self.JacobiConv
            self.alpha = self.beta = 0
        elif conv_fn_type == "chebyshev":
            self.conv_fn = self.JacobiConv
            self.alpha = self.beta = -0.5
        elif conv_fn_type == "jacobi":
            self.conv_fn = self.JacobiConv
            self.alpha = alpha
            self.beta = beta

    def forward(self, x: Tensor, adj: Tensor):
        '''
        Args:
            x: node embeddings. of shape (number of nodes, node feature dimension)
            edge_index and edge_attr: If the adjacency is cached, they will be ignored.
        '''
        thetas = [self.basetheta * torch.tanh(i) for i in self.thetas]

        xs = []
        theta_xs = []
        for L in range(self.depth+1):
            tx = self.conv_fn(L, xs, adj)
            xs.append(tx)
            theta_xs.append(thetas[L] * tx @ x)
            
        out = sum(theta_xs)
        return out


    def identity(self, adj):
        i = torch.eye(adj.shape[-1], device=adj.device)
        i = i.unsqueeze(0).unsqueeze(0)
        i = i.expand(adj.shape[0], adj.shape[1], -1, -1)
        return i


    def PowerConv(self, L, xs, adj):
        '''
        Monomial bases.
        '''
        if L == 0: 
            return self.identity(adj)
        else:
            return (adj @ xs[-1])


    def JacobiConv(self, L, xs, adj):
        if L == 0:
            return self.identity(adj)
        elif L == 1:
            return 0.5 * (self.alpha - self.beta + (self.alpha + self.beta + 2) * adj)
        else:
            A_l = ((2*L+self.alpha+self.beta) * (2*L+self.alpha+self.beta-1)) / (2*L*(L+self.alpha+self.beta))
            B_l = ((2*L+self.alpha+self.beta-1) * (self.alpha**2-self.beta**2)) / (2*L*(L+self.alpha+self.beta)*(2*L+self.alpha+self.beta-2))
            C_l = ((L+self.alpha-1)*(L+self.beta-1)*(2*L+self.alpha+self.beta)) / (L*(L+self.alpha+self.beta)*(2*L+self.alpha+self.beta-2))
            
            return (A_l * adj + B_l) @ xs[-1] - C_l * xs[-2]
