from typing import *
from typing_extensions import *

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
    return pow(var)

# Optional : ``Optional[X]`` is equivalent to ``Union[X, None]``.
def foo(a : Optional[int] = None) -> int:
    return a if a is not None else 0

# Callable : Callable[[*args, **kwargs], return]
def bar(bar_fn : Callable[[int, int], float]):
    return bar_fn(1, 2)

# Type[C] : Accept any object of class C or a subclass of C.
# all subclasses of C should implement the same constructor
# signature and class method signatures as C. The type checker
# should flag violations of this, but should also allow
# constructor calls in subclasses that match the constructor
# calls in the indicated base class.
class C : ...

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
        self.dictionary = dictionary

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
    x : float = 0
    y : float = 0
    label : str = "origin"

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
# for more constrained type checking.

def cdevide(a: int, b: int) -> int:
    return a // b

def cdevide(a: float, b: float) -> float:
    return a / b

# Notice that in the above example if we used `Union`, that is:
# ``cdevide(a: Union[int, float], b: Union[int, float])``
# it would accepts the combinations: (int, int), (int, float),
# (float, int), and (float, float). But when we use overload,
# we only accept two of the combinations: (int, int), (float, float).
# This distinction is subtle but most important for the existance of
# overload

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
