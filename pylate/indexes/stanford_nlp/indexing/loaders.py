import os
import re

import ujson


def load_doclens(directory, flatten=True):
    doclens_filenames = {}

    for filename in os.listdir(directory):
        match = re.match(r"doclens.(\d+).json", filename)

        if match is not None:
            doclens_filenames[int(match.group(1))] = filename

    doclens_filenames = [
        os.path.join(directory, doclens_filenames[i])
        for i in sorted(doclens_filenames.keys())
    ]

    all_doclens = []
    for filename in doclens_filenames:
        with open(filename) as f:
            all_doclens.append(ujson.load(f))

    if flatten:
        all_doclens = [x for sub_doclens in all_doclens for x in sub_doclens]

    if len(all_doclens) == 0:
        raise ValueError("Could not load doclens")

    return all_doclens
