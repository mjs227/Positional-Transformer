import torch
import torch.nn as nn


def dot(x, y):
    out_t = torch.sum(x * y, dim=-1)

    return torch.reshape(out_t, (out_t.shape[0], 1))


class PTLayer(nn.Module):
    def __init__(self, in_dim, window, p):
        super(PTLayer, self).__init__()
        self.IN_DIM = in_dim
        self.WINDOW = window
        self.q = nn.Linear(self.IN_DIM, self.IN_DIM)
        self.k_prev = nn.ModuleList([nn.Linear(self.IN_DIM, self.IN_DIM) for _ in range(window)])
        self.k_cent = nn.Linear(self.IN_DIM, self.IN_DIM)
        self.k_next = nn.ModuleList([nn.Linear(self.IN_DIM, self.IN_DIM) for _ in range(window)])
        self.ff_prev = nn.ModuleList([nn.Linear(self.IN_DIM, self.IN_DIM) for _ in range(window)])
        self.ff_cent = nn.Linear(self.IN_DIM, self.IN_DIM)
        self.ff_next = nn.ModuleList([nn.Linear(self.IN_DIM, self.IN_DIM) for _ in range(window)])
        self.dropout = nn.Dropout(p=p)
 
    def forward(self, x):
        x_split = torch.split(x, self.IN_DIM, dim=1)
        x_prev = list(x_split[:self.WINDOW])
        x_cent = x_split[self.WINDOW]
        x_next = list(x_split[self.WINDOW + 1:])

        q = self.q(x_cent)
        x_cent_ = x_cent * dot(q, self.k_cent(x_cent))

        for i in range(self.WINDOW):
            attn_prev_i = dot(k, self.k_prev[i](x_prev[i]))
            attn_next_i = dot(k, self.k_next[i](x_next[i]))
            x_cent_ = x_cent_ + (x_prev[i] * attn_prev_i) + (x_next[i] * attn_next_i)
            
            x_prev[i] = self.ff_prev[i](x_prev[i])
            x_prev[i] = self.dropout(x_prev[i])
            x_prev[i] = torch.nn.functional.normalize(x_prev[i], p=2, dim=1)
            
            x_next[i] = self.ff_next[i](x_next[i])
            x_next[i] = self.dropout(x_next[i])
            x_next[i] = torch.nn.functional.normalize(x_next[i], p=2, dim=1)

        x_cent_ = torch.nn.functional.normalize(x_cent_, p=2, dim=1)
        x_cent_ = self.ff_cent(x_cent_)
        x_cent_ = self.dropout(x_cent_)
        x_cent_ = torch.nn.functional.normalize(x_cent_ + x_cent, p=2, dim=1)

        return torch.cat(tuple(x_prev + [x_cent_] + x_next), dim=1)
        

class PTFinalLayer(nn.Module):
    def __init__(self, in_dim, window):
        super(PTFinalLayer, self).__init__()
        self.IN_DIM = in_dim
        self.WINDOW = window
        self.q = nn.Linear(self.IN_DIM, self.IN_DIM)
        self.k_prev = nn.ModuleList([nn.Linear(self.IN_DIM, self.IN_DIM) for _ in range(window)])
        self.k_cent = nn.Linear(self.IN_DIM, self.IN_DIM)
        self.k_next = nn.ModuleList([nn.Linear(self.IN_DIM, self.IN_DIM) for _ in range(window)])
 
    def forward(self, x):
        x_split = torch.split(x, self.IN_DIM, dim=1)
        x_prev = x_split[:self.WINDOW]
        x_cent = x_split[self.WINDOW]
        x_next = x_split[self.WINDOW + 1:]

        q = self.q(x_cent)
        attn_prev = [dot(q, self.k_prev[i](x_prev[i])) for i in range(self.WINDOW)]
        attn_cent = dot(q, self.k_cent(x_cent))
        attn_next = [dot(q, self.k_next[i](x_next[i])) for i in range(self.WINDOW)]

        attn = torch.cat(tuple(attn_prev + [attn_cent] + attn_next), dim=1)
        attn = torch.softmax(attn, dim=1)
        attn_split = torch.split(attn, 1, dim=1)

        attn_prev_ = attn_split[:self.WINDOW]
        attn_cent_ = attn_split[self.WINDOW]
        attn_next_ = attn_split[self.WINDOW + 1:]

        x_cent = x_cent * attn_cent_

        for i in range(self.WINDOW):
            x_cent = x_cent + (x_prev[i] * attn_prev_[i]) + (x_next[i] * attn_next_[i])

        return torch.nn.functional.normalize(x_cent, p=2, dim=1)
        

class PositionalTransformer(nn.Module):
    def __init__(self, d_model=128, n_layers=3, window=1, p=0.1):
        super(PositionalTransformer, self).__init__()
        self.layers = nn.ModuleList([PTLayer(d_model, window, p) for _ in range(n_layers)])
        self.final_layer = PTFinalLayer(d_model, window)
  
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        return self.final_layer(x)
