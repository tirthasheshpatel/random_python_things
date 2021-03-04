# Typing in Python

These are some personal notes on the `typing` module in python.

### Some PEPs

There are a lot of PEPs that have introduced and build upon the
existing typing module in python. Some of them are listed below:

- PEP 483 and PEP 484 : Introduces typing in Python. PEP 484 adds
                        the `typing_extensions` module in `Python <= 3.8`
                        and is merged with `typing` module in `Python 3.9`.
- PEP 563 : `from __future__ import annotations` => posponed
            evaluation of annotations. (becomes default in `3.10`.
            Useful for compatibility with `3.8` and `3.9`). Prevents
            annotations from being evaluated during runtime, thus storing
            them as plain strings
- PEP 593 : Introduces `Annotated` to provide metadata in Typing.
- PEP
