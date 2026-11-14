from torch import nn
from utils.ctc_vocab import vocab

class CTCHead(nn.Module):
    def __init__(self, embed_dim, vocab_size=len(vocab), upsample_factor=3):
        super().__init__()
        self.upsample_factor = upsample_factor

        # Deconvolution:
        # Upsamples Time frame patches using Transpose Convolution 1d
        # shape In: B x D x 187
        # shape out: B x D x (187 * factor)
        # with factor = 3, T is 562:
        # Out = (I-1) * stride + K - 2 * pad = (187-1) * 3 + 6 - 2 = 562
        self.upsample = nn.ConvTranspose1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=upsample_factor * 2,
            stride=upsample_factor,
            padding=upsample_factor // 2
        )
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x): # B x 1496 x D
        B = x.size(0)
        x = x.view(B, 8, 187, -1) # B x 8 x 187 x D
        # TODO: use attention instead of mean pooling
        x = x.mean(dim=1) # B x 187 x D
        x = x.transpose(1,2) # B x D x 187
        x = self.upsample(x) # B x D x (187*factor)
        x = x.transpose(1,2) # B x (187*factor) x D
        x = self.linear(x) # B x (187*factor) x V
        x = x.log_softmax(dim=-1) # B x T x V
        x = x.transpose(0,1) # T x B x V - required shape for CTC loss
        return x