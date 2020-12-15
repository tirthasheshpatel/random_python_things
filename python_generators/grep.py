# Simple generator examples
from collections.abc import Generator

def grep(pattern):
    print("Hello There! I am a generator. Send me something")
    while True:
        line = (yield)
        if pattern in line:
            print(line)

# Use the following script to run the code!
__doc__ = '''
Examples of Generators

Example of `grep` function

>>> from grep import grep
>>> g = grep("python")
>>> g
<generator object grep at 0x0000020AFF1CCA20>
>>> next(g)
Hello There! I am a generator. Send me something
>>> g.send('I am the best!')
>>> g.send('print something please!')
>>> g.send('I love python!')
I love python!
>>> g.send('Hmmm.... I think python is the trigger word, or is it??!')
Hmmm.... I think python is the trigger word, or is it??!

'''

def startit(generator_func):
    def _start(*args, **kwargs):
        g = generator_func(*args, **kwargs)
        if not isinstance(g, Generator):
            raise TypeError("This decorator can only used to start generators"
                            " but your function doesn't seem to be one! found"
                            " {}".format(type(generator_func)))
        g.send(None)
        return g
    return _start

@startit
def decorated_grep(pattern):
    print("Hello there! i am decorated! so you would see this"
          " as soon as you create me! please consider sending"
          " some values soon! cheers!")
    while True:
        line = (yield)
        if pattern in line:
            print(line)

__doc__ += '''
Example of `decorated_grep` function

>>> from grep import decorated_grep
>>> decorated_grep
<function startit.<locals>._start at 0x000002FA88DBBD90>
>>> g = decorated_grep("python")
Hello there! i am decorated! so you would see this as soon as you create me! please consider sending some values soon! cheers!
>>> g.send("Hi! How are you?")
>>> g.send("i love python so much!")
i love python so much!
>>> g.send("python is the best!")
python is the best!
'''
