# Coroutine Examples!!
from time import sleep
from grep import startit as coroutine

@coroutine
def printer():
    while True:
        line = (yield)
        print(line, end='')

def follow(file, target):
    file.seek(0, 2)
    while True:
        line = file.readline()
        if not line:
            sleep(0.1)
            continue
        target.send(line)

__doc__ = '''
Examples of coroutines!

Example of a simple `coroutine`

>>> from coroutines import follow, printer
>>> p = printer()
>>> p
<generator object printer at 0x0000025007A28228>
>>> file = open('logs.txt', 'r')
>>> g = follow(file, p)
Here we meet again!
How do you do that with python!!
This is amazing!


'''

@coroutine
def filter(target):
    # Converts the text into uppercase
    while True:
        line = (yield)
        line = line.upper()
        target.send(line)

__doc__ += '''
Example of 2 level pipelined coroutines. In this example, filter turns the
values sent by generator to uppercase and sends the uppercase values
to the printer. Hence, this works in a 3 level pipeline! This is what is happening!

follow (data source) -> filter (mid level coroutine) -> printer (final coroutine)

>>> from coroutines import printer, follow, filter
>>> p = printer()
>>> f = filter(p)
>>> file = open('logs.txt', 'r')
>>> g = follow(file, f)
THIS WILL FILTER ME TO UPPERCASE!!
ISN'T THAT JUST AMAZING!!
I LOVE IT...


'''

@coroutine
def grep(pattern, target):
    while True:
        line = (yield)
        if pattern in line:
            target.send(line)

__doc__ += '''
Example of 3 level coroutines

follow (data source)
       |
grep (intermediate coroutine)
       |
filter (intermediate coroutine)
       |
printer (final coroutine)

>>> from coroutines import grep, filter, follow, printer
>>> p = printer()
>>> f = filter(p)
>>> gr = grep("python", f)
>>> file = open('logs.txt', 'r')
>>> follow(file, gr)
PYTHON IS MY JAM!
ISN'T PYTHON JUST AMAZING?
PYTHON FOR LIFE!


'''

__doc__ += '''
The main difference between generator and a coroutine is
that a generator pulls data out of a pipeline and coroutine
pushes (sends) data to the pipeline!


'''

@coroutine
def named_printer(name):
    while True:
        line = (yield)
        print("{}: {}".format(name, line), end='')

@coroutine
def broadcast(targets):
    while True:
        line = (yield)
        for target in targets:
            target.send(line)

__doc__ += '''
Example of broadcasting with 2 level pipes.

                                   |--> grep("python") --> printer("python")
follow (data source) --> broadcast |--> grep("tirth")  --> printer("tirth")
                                   |--> grep("zen")    --> printer("zen")

>>> from coroutines import named_printer, broadcast, grep, follow
>>> p1 = named_printer("python")
>>> p2 = named_printer("tirth")
>>> p3 = named_printer("zen")
>>> gr1 = grep("python", p1)
>>> gr2 = grep("tirth", p2)
>>> gr3 = grep("zen", p3)
>>> b = broadcast([gr1, gr2, gr3])
>>> file = open('logs.txt', 'r')
>>> follow(file, b)
python: i am python and people love me!
tirth: i am tirth who wrote this
python: i am zen of python
zen: i am zen of python
python: tirth is a zen of python
tirth: tirth is a zen of python
zen: tirth is a zen of python


'''

__doc__ += '''
One more example of broadcasting but a little scary!

                                   |--> grep("python") --|
follow (data source) --> broadcast |--> grep("tirth")  --|--> printer
                                   |--> grep("zen")    --|

>>> from coroutines import printer, follow, grep, broadcast
>>> p = printer()
>>> gr1 = grep("python", p)
>>> gr2 = grep("tirth", p)
>>> gr3 = grep("zen", p)
>>> b = broadcast([gr1, gr2, gr3])
>>> file = open('logs.txt', 'r')
>>> follow(file, b)
python just became scary
but it is as beautiful as tirth
and he is the zen of python
and he is the zen of python
tirth is pythonic zen
tirth is pythonic zen
tirth is pythonic zen


'''

__doc__ += '''
=============================
         INTERLUDE
=============================
1. Coroutines provide more data routing possibilities
   than simple iterators
2. If you build a collection ofsimple data processing
components, you can glue them together into complex
arrangements of pipes, branches and mergeing.
3. Some limitations to be discussed later...
'''