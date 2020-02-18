class EnforceMeta(type):
    def __new__(cls, clsname, bases, clsdict):
        for name in clsdict:
            if name.lower() != name:
                raise TypeError(f"Method {name} must be in lower case.")
        return super().__new__(cls, clsname, bases, clsdict)


class Root(metaclass=EnforceMeta):
    pass


class ChildA(Root):
    def method_name(self):
        pass


class ChildB(Root):
    def methodName(self):
        pass


a = ChildA()
b = ChildB()
