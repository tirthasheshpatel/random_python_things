# TODO: add tests
from ..all_module_contents import *

function("this is anything") # valid
function(function) # valid
function(123456) # valid

try:
    # valid
    hahalol("this should throw an exception")
except HahaLolError:
    pass

try:
    print("In an infinite loop. Press Ctrl+C to continue")
    hahalol("inf_loop") # valid
except KeyboardInterrupt:
    pass

take_a_tuple(("this", "is", "a", "valid", "argument")) # valid
take_a_tuple("this is not a valid argument") # invalid

square(10)
square(10.10)
square("invalid argument")

foo()
foo(None)
foo(10)
foo("invalid argument")

bar(lambda x, y : 1.*(x + y))
def invalid_func(x : float, y : float) -> int:
    from math import floor
    return floor(x + y)
bar(invalid_func) # invalid

accept_c(C) # valid
accept_c(CC) # valid
accept_c(CCC) # valid
accept_c(CCCC) # valid
accept_c(C()) # invalid
accept_c("this is an invalid argument") # invalid

accept_a_and_return_b('a')
accept_a_and_return_b('invalid')

starship = StarShip()
StarShip.stats = {"1": "haha", "2": "lol"}
starship.stats = {} # invalid argument

Connection.TIMEOUT = 1000 # final arguments cannot be over-ridden

my_class: MyGenericClass[int, bytes] = MyGenericClass({1: b"1", 2: b"2", 3: b"3"})
my_invalid_class: MyGenericClass[str, float] = MyGenericClass({"1": 1.0, "2": 2.0, "3": 3.0})

my_class.get(1)
my_class.get("invalid argument")

concat("valid", "argument")
concat(b"valid", b"argument")
concat(1000, 2000) # invalid argument

my_employee = Employee("Tirth", 123456)
my_invalid_class = Employee(123456, "Tirth")

VALID_ID = UserID(123456)
INVALID_ID = UserID("hahalol")

xx : Point2D = {"x": 1.1, "y": 2.0, "label": "haha"}
yy : Point2D = {"x": "invalid", "y": "dictionary", "label": 3.3}

cdevide(1, 2) # valid
cdevide(1. + 2.j, 3. + 4.j) # valid
cdevide(1, 2. + 3.j) # VALID !!! ``overload`` doesn't provide multiple
                     # dispatch michanism and so type checkers will allow
                     # any combinations of the `int` and `complex`
                     # i.e : (  int   , int    ),
                     #       (  int   , complex),
                     #       (complex , int    ),
                     #       (complex , complex)
cdevide("invalid", "arguments") # invalid!!

# invalid as ``final`` methods can't be over-ridden
Base.done = (lambda aaaa : aaaa + 1)
