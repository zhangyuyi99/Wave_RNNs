from torch import nn
import torch
import numpy as np
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN_Cell(nn.Module):
    def __init__(self, n_inp, n_hid, n_ch=1, act='tanh', ksize=3, init='eye', freeze_rnn='no', freeze_encoder='no', solo_init='no'):
        super(RNN_Cell, self).__init__()
        self.n_hid = n_hid
        self.Wx = nn.Linear(n_inp, n_hid * n_ch)
        self.Wy = nn.Linear(n_hid * n_ch, n_hid * n_ch)

        if solo_init == 'yes':
            nn.init.zeros_(self.Wx.weight)
            nn.init.zeros_(self.Wx.bias)
            with torch.no_grad():
                w = self.Wx.weight.view(n_ch, n_hid, n_inp)
                w[:, 0] = 1.0
        elif solo_init == 'no':
            nn.init.normal_(self.Wx.weight, mean=0.0, std=0.001)
        else:
            raise NotImplementedError

        if init == 'eye':
            nn.init.eye_(self.Wy.weight)
            nn.init.zeros_(self.Wy.bias)
        elif init == 'fwd':
            nn.init.eye_(self.Wy.weight)
            nn.init.zeros_(self.Wy.bias)
            with torch.no_grad():
                self.Wy.weight.data = torch.roll(self.Wy.weight, 1, -1).data
        elif init =='rand':
            pass
        else:
            raise NotImplementedError

        if act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'ident':
            self.act = nn.Identity()
        else:
            raise NotImplementedError
	
        if freeze_encoder == 'yes':
            for param in self.Wx.parameters():
                param.requires_grad = False
        else:
            assert freeze_encoder == 'no'

        if freeze_rnn == 'yes':
            for param in self.Wy.parameters():
                param.requires_grad = False
        else:
            assert freeze_rnn == 'no'

    def forward(self,x,hy):
        hy = self.act(self.Wx(x) + self.Wy(hy))
        return hy


class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, n_ch=1, act='relu', ksize=3, init='eye', freeze_rnn='no', freeze_encoder='no', solo_init='no'):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        self.n_ch = n_ch
        self.spatial = int(np.sqrt(n_hid))
        self.cell = RNN_Cell(n_inp, n_hid, n_ch, act, ksize,  init, freeze_rnn, freeze_encoder, solo_init)
        self.readout = nn.Linear(self.n_hid * self.n_ch, n_out)

    def forward(self, x, get_seq=False):
        ## initialize hidden states
        hy = Variable(torch.zeros(x.size(1), self.n_hid * self.n_ch)).to(device)
        y_seq = []
        outputs = []

        for t in range(x.size(0)):
            hy = self.cell(x[t].unsqueeze(1), hy)
            if get_seq:
                y_seq.append(hy.view(x.size(1), self.n_ch, self.spatial, self.spatial).detach().cpu())
            output = self.readout(hy)
            outputs.append(output)

        if get_seq:
            y_seq = torch.stack(y_seq, dim=0)
        outputs = torch.stack(outputs, dim=0)

        return outputs, y_seq
