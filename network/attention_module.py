import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np

class Attention_module(nn.Module):
    def __init__(self, fea_dim, out_dim):
        super(Attention_module, self).__init__()
        self.fea_dim = fea_dim
        self.out_dim = out_dim

        self.WQ = Parameter(torch.Tensor(self.fea_dim, self.out_dim))
        self.WK = Parameter(torch.Tensor(self.fea_dim, self.out_dim))
        self.WV = Parameter(torch.Tensor(self.fea_dim, self.fea_dim)) 

        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.WQ)
        nn.init.xavier_uniform_(self.WK)
        nn.init.xavier_uniform_(self.WV)

        
    def forward(self, input, input_ori): 
        '''
        input : k nearest image k*2048
        input_ori : image 2048 
        '''
        # original space
        Q = torch.matmul(input, self.WQ) # original space Q K batch * k*2048
        K = torch.matmul(input, self.WK) # batch * k * 2048
        QKT = nn.Softmax(dim = -1)((torch.bmm(Q, K.permute(0,2,1)))/math.sqrt(self.out_dim)) # batch * k * k
        Z_ori = torch.bmm(QKT, input)  # batch * k * 2048

        trans_input = torch.matmul(input, self.WV) # feature adaptation
        Q2 = torch.matmul(trans_input, self.WQ)
        K2 = torch.matmul(trans_input, self.WK)
        QKT2 = nn.Softmax(dim = -1)((torch.bmm(Q2, K2.permute(0,2,1)))/math.sqrt(self.out_dim))
        Z_new = torch.bmm(QKT2, trans_input)
        Z_new += trans_input
        V_ori = (input_ori) 
        V_new = torch.matmul(input_ori, self.WV)
        return Z_ori,Z_new, V_ori,V_new

class Attention(nn.Module):
    def __init__(self, fea_dim, out_dim):
        super(Attention, self).__init__()
        self.fea_dim = fea_dim
        self.out_dim = out_dim
        self.encoder1 = self._make_encoder(self.fea_dim, self.out_dim)
    def _make_encoder(self, fea_dim, out_dim):
        return Attention_module(fea_dim, out_dim)
    def forward(self,x, input_ori):
        out, out2, V_ori, V_new = self.encoder1(x, input_ori)        
        out_mean = torch.mean(out, dim = 1)
        out2_mean = torch.mean(out2, dim = 1)
        return out_mean, out2_mean, V_ori, V_new