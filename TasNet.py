
import torch
import torch.nn as nn
from torch.linalg import norm
import torch.nn.functional as F

EPS = 1e-8



class TasNet(nn.Module):
    def __init__(self, N, L, stride, num_spk, hidden_size, num_layers, bidirectional=False):
        super(TasNet, self).__init__()
        self.N = N
        self.L = L
        self.stride = stride
        self.num_spk = num_spk
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.encoder = Encoder(N, L, stride)
        self.separator = Separator(N, num_spk, hidden_size, num_layers, bidirectional)
        self.decoder = Decoder(N, L, stride, num_spk)

    def forward(self, mixture):

        mixture_weights, norm_coef = self.encoder(mixture)
        masks = self.separator(mixture_weights)
        sources = self.decoder(mixture_weights, masks, norm_coef)

        return sources


class Encoder(nn.Module):
    def __init__(self, N, L, stride):
        super(Encoder, self).__init__()
        """
            N: the encoding dimension, i.e., the number of basis signal
            L: the segment Length
            stride: the stride size to cut the segments
        """

        self.L = L
        self.N = N
        self.stride = stride
        # output from conv1d_U and conv1d_V is [B, N, K], K is determined by mixture length, L and stride K = floor( (T-L)/stride + 1), T is the total time points of the mixture

        self.U = nn.Conv1d(1, self.N, kernel_size=self.L, stride=self.stride, bias=False)
        self.V = nn.Conv1d(1, self.N, kernel_size=self.L, stride=self.stride, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: the mixed speech [B, 1, T], T is the number of time points.
        Returns:
            mixture_weights: [B, N, K], K is determined by mixture length, L and stride K = floor( (T-L)/stride + 1)
        """
        norm_coef = mixture.norm(p = 2, dim = 2, keepdim=True)
        mixture_normalized = mixture/(norm_coef + EPS)
        #mixture_normalized = F.normalize(mixture, p = 2, dim = 2)

        mixture_weights = F.relu(self.U(mixture_normalized))*torch.sigmoid(self.V(mixture_normalized)) # [B, N, K]

        return mixture_weights, norm_coef

class Separator(nn.Module):
    def __init__(self, N, num_spk, hidden_size, num_layers, bidirectional=False):
        super(Separator, self).__init__()
        self.N = N
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_spk = num_spk
        self.bidirectional = bidirectional

        # Components
        self.layer_norm = nn.LayerNorm(self.N)
        self.rnn = nn.ModuleList([nn.LSTM(self.N, self.hidden_size, num_layers=1, batch_first=True, bidirectional = self.bidirectional)])
        # for skip connection starting from second layer
        for i in range(1, self.num_layers):
            self.rnn.append(nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True, bidirectional = self.bidirectional))

        rnn_out_dim = self.hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(rnn_out_dim, self.N*self.num_spk)

    def forward(self, mixture_weights):
        """
        Args:
            mixture_weights: output from encoder layer, size is [B, N, K], K is the number of segments. N is the encoded dimension
        Return:
            masks: [B, K, num_spk, N]
        """
        # transpose to [B, K, N] for layer norm and LSTM, K is along the time axis, N is the encoded dimension.
        mixture_weights = mixture_weights.transpose(1,2)
        norm_mixture_weights = self.layer_norm(mixture_weights)

        input = norm_mixture_weights
        rnn_output, _ = self.rnn[0](input)
        input = rnn_output
        for i in range(1, self.num_layers):
            # Skip connection
            skip_input = input
            rnn_output, _ = self.rnn[i](input)
            input = rnn_output + skip_input

        fc_output = self.fc(input)
        masks = F.softmax(fc_output, dim=2)
        # Transpose back to make K (the number of segment) the last dimension
        masks = masks.transpose(1, 2) #[B, num_spk*N, K]
        return masks


class Decoder(nn.Module):
    def __init__(self, N, L, stride, num_spk):
        super(Decoder, self).__init__()
        self.N = N
        self.L = L
        self.stride = stride
        self.num_spk = num_spk
        # The transposed convolution
        self.transposeConv = nn.ConvTranspose1d(self.N, 1, self.L, bias=False, stride=self.stride)


    def forward(self, mixture_weights, mask, norm_coef):
        """
        Args:
            mixture_weights: [B, N, K] the initial weights calculated by encoder
            mask: [B, num_spk * N, K] the mask for each speaker
            norm_coef: [B, 1, 1]the normalize factor used by encoder
        Return:
            sources: [B, num_spk, T], T is the total time point of the signal
        """
        # Prepare mask for decoding
        batch_size = mask.size(0)
        num_segment = mask.size(2)
        mask = mask.reshape(batch_size * self.num_spk, self.N, num_segment)
        sources = self.transposeConv(mask) #[B*num_spk, 1, T]
        sources = sources.view(batch_size, self.num_spk, -1)
        # reverse L2 norm
        sources = sources*norm_coef  #[B, num_spk, T]
        return sources





if __name__ == "__main__":
    N = int(512)
    sr = 8000
    x = torch.rand([2, 1, 8000])

    L = int(sr * 4/1000)
    stride = L//2

    # Test encoder
    encoder = Encoder(N, L, stride)
    weight, norm_coef = encoder(x)
    print(weight.shape)
    print(norm_coef.shape)

    hidden_size = 1000
    num_layers = 4
    num_spk = 2
    bidirectional = False
    # Test Separator
    separator = Separator( N, num_spk, hidden_size, num_layers, bidirectional=bidirectional)
    masks = separator(weight)
    print(masks.shape)

    decoder = Decoder(N, L, stride, num_spk)
    sources = decoder(weight, masks, norm_coef)
    print(sources.shape)


    # Test TasNet pad signal
    x = torch.rand([2, 1, 7999])
    print(x.type())

    tasnet = TasNet(N, L, stride, num_spk, hidden_size, num_layers, bidirectional)
    sources = tasnet(x)
    print(sources.shape)

