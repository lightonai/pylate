import tqdm

__all__ = ["iter_batch"]


def iter_batch(
    X: list[str], batch_size: int, tqdm_bar: bool = True, desc: str = ""
) -> list:
    """Iterate over a list of elements by batch."""
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
