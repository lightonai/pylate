collect_ignore = []

try:
    import voyager  # noqa: F401
except ImportError:
    collect_ignore.append("pylate/indexes/voyager.py")
