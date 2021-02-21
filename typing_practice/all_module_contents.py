__all__ : list = [
    'HahaLolError',
    'function',
    'hahalol',
    'take_a_tuple',
    'square',
    'foo',
    'bar',
    'C', 'CC', 'CCC', 'CCCC',
    'accept_c',
    'accept_a_and_return_b',
    'StarShip',
    'Connection',
    'MyGenericClass',
    'concat',
    'Proto',
    'B',
    'proto_func',
    'Employee',
    'UserID',
    'Point2D',
    'cdevide',
    'Base',
    'Sub'
]

from sys import version_info

from typing import *
if version_info <= (3, 8):
    from typing_extensions import * # type: ignore

# Setup Code
class HahaLolError(Exception):
    ...

# Any : any python construct...
def function(a : Any) -> Any:
    return "haha", 4, 4, 5

# NoReturn : Used for a function/method that never
#            terminates or always throws exception.
def hahalol(a : str) -> NoReturn:
    if a == "inf_loop":
        while 1: ...
    else:
        raise HahaLolError("i will always raise!!!!!")

# Tuple : tuple object
def take_a_tuple(a : Tuple) -> Tuple:
    return a

# Union : A union of two python constructs
def square(var : Union[int, float]) -> Union[int, float]:
    from math import pow
    return pow(var, 2)

# Optional : ``Optional[X]`` is equivalent to ``Union[X, None]``.
def foo(a : Optional[int] = None) -> int:
    return a if a is not None else 0

# Callable : Callable[[*args, **kwargs], return]
def bar(bar_fn : Callable[[int, int], float]):
    return bar_fn(1, 2)

# Type[C] : A variable annotated with C may accept a value
# of type C. In contrast, a variable annotated with Type[C]
# may accept values that are **classes themselves**.
# all subclasses of C should implement the same constructor
# signature and class method signatures as C. The type checker
# should flag violations of this, but should also allow
# constructor calls in subclasses that match the constructor
# calls in the indicated base class.
class C : ...
class CC(C): ...
class CCC(CC): ...
class CCCC(CCC): ...

# accepts all C, CC, CCC, CCCC
def accept_c(cls : Type[C]) -> str:
    return "accepted"

# Literal : A function accepting or returning a fixed set
# of values or a fixed value can be represented by a `Literal`.
def accept_a_and_return_b(a : Literal['a']) -> Literal['b']:
    return 'b'

# ClassVar : ClassVar indicates that a given attribute is
# intended to be used as a class variable and should not be
# set on instances of that class.
# {{See the documentation page for more details}}
class StarShip :
    stats : ClassVar[Dict[str, str]] = {}
    damage : int = 10

# Final : A special construct to indicate to type checkers that
# a name cannot be re-assigned or overridden in a subclass.
class Connection :
    TIMEOUT : Final[int] = 10

# Annotated : I did not understand. So, skipping...
# Probably, this construct is esoteric and not yet
# adopted by more catholic libraries...
...

# Generic : A generic type basically binds a `TypeVar`
# (or other user defined types) with a class such that
# the TypeVar variable can be used anywhere in the class.
# Also remember that, the class can be indexed to pass in
# the type for the class TypeVar. For example, we can do:
# `` MyGenericClass[int, int] ``. So, we set U=int, V=int.
U = TypeVar("U", int, float)
V = TypeVar("V", str, bytes)

class MyGenericClass(Generic[U, V]):
    def __init__(self, dictionary : Mapping[U, V]) -> None:
        self.dictionary : Mapping[U, V] = dictionary

    def get(self, key : U) -> V:
        return self.dictionary[key]

# AnyStr : AnyStr is defined as `TypeVar("AnyStr", str, bytes)`
def concat(a : AnyStr, b : AnyStr) -> AnyStr :
    return a + b

# Protocol : Protocol classes are just used to indicate the type
# checker the methods, class variables, etc available for some
# parameter that is passed to the function.
# We can also pass TypeVar as arguments to the `Protocol`. For example
# ``Protocol[T]`` takes a TypeVar T and binds to the class. We are free
# to use the typevar T anywhere inside the class.
class Proto(Protocol):
    def meth(self) -> int: ...

class B :
    def meth(self): return 0

def proto_func(x : Proto) -> int:
    return x.meth()

# runtime_checkable : esoteric --> see the documentation...
...

# NamedTuple : Typed version of collections.namedtuple
class Employee(NamedTuple):
    name : str
    id   : int
# Equivalent to:
# ``Employee = collections.namedtuple('Employee', ['name', 'id'])``
# See some more details in the documentation.

# NewType : Define your own type!
UserID = NewType("UserID", int)

# TypedDict : Add type hints to a dictionary
class Point2D(TypedDict):
    x : float
    y : float
    label : str

# Generic Concrete Collections
# ============================
# Dict
# List
# Tuple
# Set
# FrozenSet
# DefaultDict
# OrderedDict
# ChainMap
# Counter
# Deque
# IO, TextIO, BinaryIO
# Pattern, Match
# Text

# Some Important ABCs
# ===================
# Iterable
# Mapping
# Sequence
# Iterator
# Generator
# Hashable
# Sized

# Some Important Protocols
# ========================
# SupportsAbs
# SupportsBytes
# SupportsComplex
# SupportsFloat
# SupportsIndex
# SupportsInt

# Some important decorators
# =========================
# cast
# overload
# final
# no_type_check
# type_check_only

# overload : If some funtion takes a specific combination
# of parameters then we can use `overload` (rather than `Union`)
# for more constrained type checking. See the example below.

# Note : Documentation page isn't very helpful for understading the overload
# decorator. See PEP-484 instead:
# https://www.python.org/dev/peps/pep-0484/#function-method-overloading


# Some Notes on the `overload` decorator from PEP-484
# ===================================================
# FIRST POINT TO NOTE: Union cannot express the relationship between
#                      the argument and the return type, ``overload``
#                      can!
# 
# MORE DESCRIPTION OF THE FIRST POINT.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The @overload decorator allows describing functions and methods that
# support multiple different combinations of argument types. This pattern
# is used frequently in builtin modules and types. For example, the
# ``__getitem__()`` method of the bytes type can be described as follows:
# -------------------------------------------------------
# from typing import overload
# 
# class bytes:
#     ...
#     @overload
#     def __getitem__(self, i: int) -> int: ...
#     @overload
#     def __getitem__(self, s: slice) -> bytes: ...
# ------------------------------------------------------
# This description is more precise than would be possible using unions
# which cannot express the relationship between the argument and return
# types.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# POINT TWO TO NOTE: ``overload`` doesn't provide a multiple dispatch
#                    implementation. So, overloading a function with
#                    (int, int) -> int and (float, float) -> float
#                    would still accept (int, float) and (float, int)
#                    combinations of arguments. An example is given in
#                    the description.
# 
# # MORE DESCRIPTION OF THE SECOND POINT.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# While it would be possible to provide a multiple dispatch implementation
# using this syntax, its implementation would require using sys._getframe(),
# which is frowned upon. Also, designing and implementing an efficient
# multiple dispatch mechanism is hard, which is why previous attempts were
# abandoned in favor of functools.singledispatch(). (See PEP 443, especially
# its section "Alternative approaches".) In the future we may come up with a
# satisfactory multiple dispatch design, but we don't want such a design to
# be constrained by the overloading syntax defined for type hints in stub
# files. It is also possible that both features will develop independent
# from each other (since overloading in the type checker has different use
# cases and requirements than multiple dispatch at runtime -- e.g. the
# latter is unlikely to support generic types).
# 
# A constrained ``TypeVar`` type can often be used instead of using the
# @overload decorator. For example, the definitions of `concat1` and
# `concat2` in this stub file are equivalent:
# ------------------------------------------------------
# from typing import TypeVar
# 
# Number = TypeVar('Number', int, float)
# 
# def concat1(x: Number, y: Number) -> Number: ...
# 
# @overload
# def concat2(x: int, y: int) -> int: ...
# @overload
# def concat2(x: float, y: float) -> float: ...
# ------------------------------------------------------
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@overload
def cdevide(a: int, b: int) -> int: ...
@overload
def cdevide(a: complex, b: complex) -> complex: ...
def cdevide(a, b):
    return a // b if isinstance(a, int) else a/b

# final : indicates the type checker that the method/function can't be
# overridden and, if used with class, the class cannot be subclassed.
class Base :
    @final
    def done(self) -> None:
        ...

@final
class Sub :
    def done(self) -> None:
        ...

# no_type_check : Decorator to indicate that annotations are not type hints.
# This works as class or function decorator. With a class, it applies
# recursively to all methods defined in that class (but not to methods
# defined in its superclasses or subclasses).
...

# type_check_only : Decorator to mark a class or function to be unavailable
# at runtime.
# TODO: didn't understand properly. See more examples and read docs again...
...
