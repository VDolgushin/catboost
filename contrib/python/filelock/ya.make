# Generated by devtools/contrib/pypi-import.

PY23_LIBRARY()



VERSION(3.2.0)

LICENSE(PD)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    filelock/__init__.py
    filelock/_api.py
    filelock/_error.py
    filelock/_soft.py
    filelock/_unix.py
    filelock/_util.py
    filelock/_windows.py
    filelock/version.py
)

RESOURCE_FILES(
    PREFIX contrib/python/filelock/
    .dist-info/METADATA
    .dist-info/top_level.txt
)

END()
