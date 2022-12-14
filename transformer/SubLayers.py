''' Define the sublayers in encoder/decoder layer '''
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Litao Yu"


class HorizontalAttention(nn.Module):

    def __init__(self, d_model, d_v, d_att):
        super().__init__()
        self.w1 = nn.Linear(d_v, d_att)
        self.w2 = nn.Linear(d_model, d_att)
        self.relu = nn.ReLU()
        self.full_att = nn.Linear(d_att, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, mh, residual):
        '''
        mh (multi-head output): bz x n_head x lq x dv
        residual: bz x lq x dq
        '''
        att1 = self.w1(mh) #bz x n_head x lq x d_att
        att2 = self.w2(residual) #bz x lq x d_att
        att = self.relu(att1+att2.unsqueeze(1)) #bz x n_head x lq x d_att
        att = self.full_att(att) # bz x n_head x lq x 1
        alpha = self.softmax(att)
        return alpha * mh

class VerticalAttention(nn.Module):
    def __init__(self, d_model, d_att):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_att)
        self.w2 = nn.Linear(d_model, d_att)
        self.relu = nn.ReLU()
        self.full_att = nn.Linear(d_att, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, residual, x):
        '''
        residual: bz x lq x dq
        x: bz x lq x d_model
        '''
        att1 = self.w1(residual) # bz x lq x d_att
        att2 = self.w2(x) # bz x lq x d_att
        att = self.relu(att1+att2)
        att = self.full_att(att) # bz x lq x d_model
        beta = self.sigmoid(att) #bz x lq x d_model
        return beta * x

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, h_attn=False, v_attn=False, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.sdp = ScaledDotProductAttention(temperature=d_k**0.5)
        self.h_attn = h_attn
        self.v_attn = v_attn
        if h_attn:
            self.horizontal_attn = HorizontalAttention(d_model, d_v, d_v)
        if v_attn:
            self.vertical_attn = VerticalAttention(d_model, d_model//4)    
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        x = self.sdp(q, k, v, mask=mask)
        if self.h_attn:
            x = self.horizontal_attn(x, residual)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        x = x.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        x = self.fc(x)
        if self.v_attn:
            x = self.vertical_attn(residual, x)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x
