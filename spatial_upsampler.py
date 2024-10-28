import torch.nn as nn
import torch


# TODO: do this eventually, not needed though
class SpatialUpsampler(nn.Module):
    def forward(self, x) -> torch.Tensor:
        # x = blerp(x)
        # x = encoder(x)
        # x = transformer(x + noise)
        # x = decoder(x)
        return x
