from sys import version_info

FUTURE_COND = version_info.major == 3 and version_info.minor < 9

# simple example
def greeting(name : str) -> str :
	return f'Hello {name}'

# type aliasing
from typing import List

Vector = List[float]

def scale(scalar : float, vector : Vector) -> Vector :
	return [scalar * num for num in vector]

# typing module
if FUTURE_COND:
	from typing import Sequence
else:
	from collections.abc import Sequence
from typing import Dict, Tuple

# alias some types.
ConnectionOptions = Dict[str, str]
Address = Tuple[str, str]
Server = Tuple[Address, ConnectionOptions]

def broadcast_message(message : str, servers : Sequence[Server]) -> None:
	...

# NewType --> Define your own type...
from typing import NewType

UserId = NewType("UserId", int)
some_id = UserId(524313)

def get_user_name(user_id : UserId) -> str:
	return "not found!"

# typechecks
user_a = get_user_name(UserId(42351))

# does not typecheck -> int is not a UserId
# mypy throws error : expected UserId, got int
user_b = get_user_name(-1) # type: ignore
# user_b = get_user_name(UserId(-1))
# The above line will succeed on mypy

# you can perform all the operations of a `int`
# on a `UserId` but the result will always be of
# type `int`.
out = UserId(12345) + UserId(67890)

# type checks are only enforced by the static type
# checker. At runtime, the statement
# ``Derived = NewType('Derived', Base)``
# will make `Derived` a **function** that immmediately
# returns **whatever** parameter you pass it. Meaning,
# the expression Derived(some_value) does not create a
# new class or introduce any overhead beyond that of a
# normal function call.

# Another way to state this is that the following exression:
# 		``some_value is Derived(some_value)``
# is always `True` at runtime.

# Uncomment the following line to get a error
# class AdminUserId(UserId): pass
# It throws a error because `UserId` is not a
# class.

# Creating `NewType` based on a derived `NewType`
ProUserId = NewType('ProUserId', UserId)

# Callable
from typing import Any
if FUTURE_COND:
	from typing import Callable
else:
	from collections.abc import Callable

def fedder(get_next_item : Callable) -> Any:
	return get_next_item()

# we can also pass the datatype of the parameter
# that the callable accepts and its return type
def make_query(query_func : Callable[[], str]) -> str:
	return "Query Results : " + query_func()

# another example of Callable with parameters and
# the return type
def async_query(on_success : Callable[[int], None],
                on_error   : Callable[[int, Exception], None]) -> None:
	try:
		on_success(100)
	except Exception as err:
		on_error(100, err)

# Generics
if FUTURE_COND:
	from typing import Mapping
else:
	from collections.abc import Mapping

Employee = NewType("Employee", str)
Position = NewType("Position", str)

def notify_by_email(employees : Sequence[Employee], \
                    emp_data  : Mapping[Employee, Position]) \
                    -> List[Position]:
	positions : List[Position] = []
	for emp in employees:
		positions.append(emp_data[emp])
	return positions

# Ever did `template<typename T>` in C++. Well, you
# are about to be stupified. All hail TypeVar('T')
from typing import TypeVar

T = TypeVar('T')

def first(l : Sequence[T]) -> T:
	return l[0]


# User Defined Generic Types
from typing import Generic
from logging import Logger

class LoggedVar(Generic[T]) :
	def __init__(self, value : T, name : str,
                     logger : Logger) -> None:
		self.value = value
		self.name = name
		self.logger = logger

	def set(self, new : T) -> None:
		self.logger.info('Set ' + repr(self.value))
		self.value = new

	def get(self) -> T:
		self.logger.info('Get ' + repr(self.value))
		return self.value

	def log(self, message : str) -> None:
		self.logger.info('%s : %s', self.name, message)

if FUTURE_COND:
	from typing import Iterable
else:
	from collections.abc import Iterable

# The `Generic` base class has `__class_getitem__` so that
# `LoggedVar[T]` is valid as a type.
def izeros(vars : Iterable[LoggedVar[int]]) -> None:
	for var in vars:
		var.set(0)

# We can also constraint the types that a `TypeVar`
# can take by doing something like this:
S = TypeVar('S', int, str)

class StrangePair(Generic[T, S]):
	...

# each type variable argument to `Generic` must be distinct.
# So, we can't do : ``class InvalidPair(Generic[T, T])``

# We can also use multiple inheritence with `Generic`
if FUTURE_COND:
	from typing import Sized
else:
	from collections.abc import Sized

# Sized : anything that contains ``__len__``
class LinkedList(Sized, Generic[T]):
	...

# When inheriting from generic classes, some type variables
# can be fixed.
class MyDict(Mapping[str, T]):
	...


# Just `Iterable` is equivalent to `Iterable[Any]`
class MyIterable(Iterable):
	...


from typing import Union

U = TypeVar('U')
Response = Union[Iterable[U], int]

# Now we can do something like : `Response[str]`
# which will be equal to : `Union[Iterable[str], int]`
def response(query : str) -> Response[str]:
	try:
		return 10*[query]
	except:
		return -1

X = TypeVar('X', int, float, complex)
Vec = Iterable[Tuple[X, X]]

def dot(v: Vec[X]) -> X:
	return sum(x*y for x,y in v)
