import tqdm


def iter_batch(
    X: list[str], batch_size: int, tqdm_bar: bool = True, desc: str = ""
) -> list:
    """Iterate over a list of elements by batch.

    Examples
    -------
    >>> from pylate import utils

    >>> X = [
    ...  "element 0",
    ...  "element 1",
    ...  "element 2",
    ...  "element 3",
    ...  "element 4",
    ... ]

    >>> n_samples = 0
    >>> for batch in utils.iter_batch(X, batch_size=2):
    ...     n_samples += len(batch)

    >>> n_samples
    5

    """
    batchs = [X[pos : pos + batch_size] for pos in range(0, len(X), batch_size)]

    if tqdm_bar:
        for batch in tqdm.tqdm(
            iterable=batchs,
            position=0,
            total=1 + len(X) // batch_size,
            desc=desc,
        ):
            yield batch
    else:
        yield from batchs
