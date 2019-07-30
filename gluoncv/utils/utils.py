__all__ = ['split_and_load']


def split_and_load(data, ctx_list, batch_axis=0, even_split=True):
    """Splits an NDArray into `len(ctx_list)` slices along `batch_axis` and loads
    each slice to one context in `ctx_list`.

    Parameters
    ----------
    data : NDArray
        A batch of data.
    ctx_list : list of Context
        A list of Contexts.
    batch_axis : int, default 0
        The axis along which to slice.
    even_split : bool, default True
        Whether to force all slices to have the same number of elements.

    Returns
    -------
    list of NDArray
        Each corresponds to a context in `ctx_list`.
    """

    if len(ctx_list) == 1:
        return [d.as_in_context(ctx_list[0]) for d in data]

    size = len(data)
    num_slice = len(ctx_list)
    step = size // num_slice
    for i in range(num_slice):
        for k in range(i*step, (i+1)*step):
            data[k].as_in_context(ctx_list[i])
    return data
