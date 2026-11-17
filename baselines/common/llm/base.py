import contextlib

import torch
from torch import nn

from ..models import MatrixFactorization


class Rec2Base(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def maybe_autocast(self, dtype=torch.float16):
        if self.device != torch.device('cpu'):
            return torch.amp.autocast('cuda', dtype=dtype)
        return contextlib.nullcontext()

    @classmethod
    def init_rec_encoder(cls, rec_model, config, precision):
        if rec_model != 'MF':
            raise ValueError(rec_model)
        return MatrixFactorization(config)


def disabled_train(self, mode=True):
    return self
