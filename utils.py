import torch


def prepare_signal(signal, L, stride):
    """
    Make sure the signal can fit the tasnet, i.e., make the signal shape to be [B, 1, T]: B is the batch size, T is the number of time points which makes the number of segment an integer based on L and stride
    """
    if signal.dim() not in [2, 3]:
        raise RuntimeError("signal can only be 2 or 3 dimensional")

    if signal.dim() == 2:
        signal = signal.unsqueeze(1)
    batch_size = signal.size(0)
    nMix = signal.size(1)
    nsample = signal.size(2)
    rest_length = (nsample - L)%stride
    if rest_length > 0:
        # pad the signal to be a multiple of the stride
        pad = torch.zeros([batch_size, nMix, stride-rest_length]).type( signal.type())
        signal = torch.cat([signal, pad], 2)

    # Zero pad both ends to make sure all samples are used the same times
    pad_end = torch.zeros([batch_size, nMix, stride]).type( signal.type())
    signal = torch.cat([pad_end, signal, pad_end], 2)
    return signal


