from grep import startit

__doc__ = '''
Example of `countdown` with generators (not a good practice)

This is not a good practice!
This example helps see a clear distiction 
between generators and couroutines.
1. Generators are "producers" of data
2. Coroutines are "consumers" of data
3. To keep your brain from exploding,
   don't mix the two concepts together.
4. Coroutines are not releted to iteration!
5. 
It is more appropriate to use coroutines here
as this function utilizes (consumes) as new
value of n to reset the counter!

'''

@startit
def countdown(n):
    print("you can reset me by sending the new value")
    while n >= 0:
        newval = (yield n)
        if newval is not None:
            n = newval
        else:
            n -= 1

__doc__ += '''
>>> from flaky import countdown
>>> g = countdown(10)
you can reset me by sending the new value
>>> g
<generator object countdown at 0x000001B12A539228>
>>> for i in g:
...  print(i)
...  if i == 5:
...   g.send(3)
...
9
8
7
6
5
3
2
1
0
'''
