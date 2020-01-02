import numpy as np


def alignment_score(alignments):
    """
    Measures how certain is the attention module.
    Args:
        alignments (torch.Tensor): batch of alignments.
    Shape:
        alignments : batch x decoder_steps x encoder_steps
    """
    return alignments.max(dim=1)[0].mean(dim=1).mean(dim=0).item()


def alignment_diagonal_score(alignments):
    """
    Distance of the given alignment matrix to diagonal alignment.
    Args:
        alignments (torch.Tensor): batch of alignments.
    Shape:
        alignments : batch x decoder_steps x encoder_steps
    """
    return abs(np.array(range(alignments.shape[1])) - alignments.argmax(dim=2)[0].numpy()).sum() / alignments.shape[1]
