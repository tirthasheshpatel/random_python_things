from collections import namedtuple

MyTuple = namedtuple("MyTuple", ('my', 'result'))

def func(x : int) -> MyTuple:
    return MyTuple(10, 20)
