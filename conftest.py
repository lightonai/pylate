collect_ignore = []

try:
    import voyager  # noqa: F401
except ImportError:
    collect_ignore.append("pylate/indexes/voyager.py")

try:
    import scann  # noqa: F401
except ImportError:
    collect_ignore.append("pylate/retrieve/xtr.py")
