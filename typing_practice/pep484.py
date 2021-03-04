# This script exploits the PEP-484 document.

# 1. Return type of `__init__` must be annotated with
#    `None` as, by default, it is `Any`.
# 2. Also note that the first argument is annotated as
#    having the enclosing class as its type.
# 3. For `classmethod`s, the first argument is annotated
#    as `type`.
class A:
    def __init__(self, a: int) -> None:
        self.a = a


# Type hints may be built-in classes (including those
# defined in standard library or third-party extension
# modules), abstract base classes, types available in
# the types module, and user-defined classes (including
# those defined in the standard library or third-party
# modules).
class B: # `B` is a valid type hint. We can do `def f(b: B)`
    ...


# anything that is acceptable as a type hint is acceptable
# in a type alias
from typing import TypeVar

T = TypeVar('T', A, B) # valid as A and B are user-defined classes.

# It is possible to declare the return type of a callable
# without specifying the call signature by substituting a
# literal ellipsis
from typing import Callable

def some_func(other_func : Callable[..., str], *args) -> Callable[..., str]:
    return other_func

# NOTE:
#     Since using callbacks with keyword arguments is not
#     perceived as a common use case, there is currently
#     no support for specifying keyword arguments with
#     `Callable`. Similarly, there is no support for
#     specifying callback signatures with a variable number
#     of arguments of a specific type.

######################################################################
##################### Notes on Generic Types #########################
######################################################################

AnyStr = TypeVar('AnyStr', str, bytes)

# We can use AnyStr to constraint the type of argument
# In particular, we can do something like a multiple dispatch.
# Meaning, use the generic types to constraint types of
# all the arguments. For example,
def concat(x : AnyStr, y : AnyStr) -> AnyStr:
    return x + y
# The function concat can be called with either two `str`
# arguments or two `bytes` arguments, but NOT with a mix
# of `str` and `bytes` arguments.

# Now, say we have a class encapsulating the str class.
class MyStr(str): ...

_ = concat(MyStr('Hello, '), MyStr('World!')) # valid
# The above call is valid as the call to `MyStr` returns
# a `str` and so a `str` is returned by the function
# `concat` instead of `MyStr`. See yourself
print("Calling concat on MyStr returns `%s`." % _.__class__)

######################################################################


######################################################################
################### Scoping Rules for `TypeVar` ######################
######################################################################
from typing import List, Generic

S = TypeVar('S')

def a_fun(x : T) -> None:
    # this is OK.
    y = [] # type: List[T]
    # this is NOT
    z = [] # type: List[S]
           #            |----> You cannot use `S` as a
           #                   global variable. It is not
           #                   allowed until `S` is defined
           #                   inside the function. For
           #                   example, having arguments of
           #                   types S or creating a TypeVar
           #                   inside the function.

class FOO(Generic[T]):
    # this is OK
    y = [] # type: List[T]
    z = [] # type: List[S]
           #            |----> Not valid for the same reason
           #                   as above!

class BAR(Generic[T]):
    # this is an error.
    something = [] # type: List[S]
    # but this is valid. We don't need the
    # type to be defined in the class to use
    # in some method.
    def some_meth(x : S) -> S:
        return x

# Also, it is not valid to use a TypeVar in a class inside a function
# even if the TypeVar is defined somewhere inside the function.
def func(x : T) -> None:
    # this is OK.
    y = [] # type: List[T]

    # this is not valid!
    class SomeClass(Generic[T]): ...
    # This also holds for a class inside of a class.
    # A generic class nested in another generic class
    # cannot use same type variables. The scope of the
    # type variables of the outer class doesn't cover
    # the inner one. See PEP-484 document for details.

######################################################################

######################################################################
# A wierd syntax that I did not know existed before reading this PEP #
######################################################################
from typing import DefaultDict

# Did you know that you can add square brackets
# to specify the type ?? Honestly, this looks quite
# ugly. But it's a funny syntax, haha...
data = DefaultDict[int, bytes]()

# Note that one should not confuse static types and
# runtime classes. The type is still erased in this
# case and the above expression is just a shorthand
# for:
import collections
data_2 = collections.defaultdict()  # type: DefaultDict[int, bytes]

######################################################################

######################################################################
############## Arbitrary Generic Types as Base Classes ###############
######################################################################
from typing import Dict, Optional

class Node: ...

# SymbolTable is a subclass of `dict` and a subtype of
# ``Dict[str, List[Node]]``.
class SymbolTable(Dict[str, List[Node]]): ...

######################################################################

######################################################################
################ Type Variables with a Upper Bound. ##################
######################################################################

from typing import Sized

# We can pass a argument `bound` to the `TypeVar` function
# to constraint that typed variable to be a subtype of the
# `bound` type. See the example below:

ST = TypeVar('ST', bound=Sized)

def longer(x: ST, y: ST) -> ST:
    if len(x) > len(y):
        return x
    else:
        return y

longer([1], [1, 2]) # valid as `List` is subtype of Sized.
longer({1}, {1, 2}) # valid as `Set` is subtype of Sized.
longer([1], {1, 2}) # valid as `List` and `Set` are subtypes of Sized.

######################################################################

######################################################################
################## Covariance and Contravariance #####################
######################################################################

# These are two new words and so I can't explain each
# of them here. Please see the PEP-484 document to
# understand them in case you forgot or don't know
# about them.

# By default, generic types are considered `invariant`
class CC: ... # a class CC
class CCC(CC): ... # a subclass of CC called CCC.

# Invariace property means that even though the class
# CCC is a subclass of CC, the type annotation List[CC]
# will strictly accept only objects of type CC and not CCC!
def invariance_property(x : List[CC]) -> None: ...

invariance_property([CC(), CC()]) # valid
invariance_property([CCC(), CCC()]) # invalid

# We can pass `covariance=True` or `contravariance=True`
# in the `TypeVar` function if either of the properties
# is desired in a class using a typed variable.
# Of course, at most one of these may be passed.

# The read-only collection classes in typing are all
# declared covariant in their type variable
# (e.g. `Mapping` and `Sequence`). The mutable
# collection classes (e.g. `MutableMapping` and
# `MutableSequence`) are declared invariant. The one
# example of a contravariant type is the `Generator`
# type, which is contravariant in the `send()` argument
# type.

######################################################################




