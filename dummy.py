import inspect

class Dummy:
    @classmethod
    def _get_param_names(cls):
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        print(f"object.__init__ = {object.__init__}")
        print(f"init = {init}")
        if init is object.__init__:
            print("same!")

        init_signature = inspect.signature(init)
        print(f"init_signature.parameters.values() =  {init_signature.parameters.values()}")
        print(f"kind = {[p.kind for p in init_signature.parameters.values()]}")
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        print(f"filtered parameters = {parameters}")
        print(f"kind of filtered parameters = {[p.kind for p in parameters]}")
        print(f"final parameters = {sorted([p.name for p in parameters])}")

    def get_params(self, deep = True):
        out = dict()
        params = self._get_param_names
        for key in params:
            try:
                value = getattr(self, key)
            except AttributeError:
                print(f"WARNING: Only instance variables are returned whenever get_params is called: ignoring {key}")
                value = None
            if deep and hasattr(value, 'get_params'):
                print("It has attribute and the method is deep!")
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k,val in deep_items)
            out[key] = value
        return out


class DummyExt(Dummy):
    def __init__(self, a, b, c, d, *args, **kwargs):
        self.a = a
        self.b = b
        self.c = c
        self.d = d 
        self.e = args
        self.f = kwargs
    
    def get_params(self):
        pass

obj = DummyExt(a = 1, b = 2, c = 3, d = 4, e = 5, f = 6)
obj._get_param_names()
