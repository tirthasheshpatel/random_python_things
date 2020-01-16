class NullInstance(type):
    def __call__(self, *args, **kwargs):
        raise TypeError("I am the DANGER!")

class PrintOnCreation(type):
    def __call__(self, *args, **kwargs):
        print("I fucked ted!")

class BreakingBad(metaclass=NullInstance):
    def __init(self, characters):
        self.characters = characters 


show = BreakingBad(["white", "jesse", "skyler"])
