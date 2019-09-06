import numpy as np


def _pad_data(x, length):
    _pad = 0
    assert x.ndim == 1
    return np.pad(
        x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])


def _pad_tensor(x, length):
    _pad = 0
    assert x.ndim == 2
    x = np.pad(
        x, [[0, 0], [0, length - x.shape[1]]],
        mode='constant',
        constant_values=_pad)
    return x


def prepare_tensor(inputs, out_steps):
    max_len = max((x.shape[1] for x in inputs)) + 1  # zero-frame
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_tensor(x, pad_len) for x in inputs])


def _pad_stop_target(x, length):
    _pad = 1.
    assert x.ndim == 1
    return np.pad(
        x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def prepare_stop_target(inputs, out_steps):
    max_len = max((x.shape[0] for x in inputs)) + 1  # zero-frame
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_stop_target(x, pad_len) for x in inputs])


def pad_per_step(inputs, pad_len):
    return np.pad(
        inputs, [[0, 0], [0, 0], [0, pad_len]],
        mode='constant',
        constant_values=0.0)


def pad_list(inputs, pad_value):
    """Perform padding for the list of tensors.
    Args:
        inputs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.
    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).
    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])
    """
    n_batch = len(inputs)
    max_len = max(x.size(0) for x in inputs)
    pad = inputs[0].new(n_batch, max_len, *inputs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, :inputs[i].size(0)] = inputs[i]

    return pad
