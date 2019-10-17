class Singleton(type):
    def __init__(self, *args, **kwargs):
        self.instance = None 
        super().__init__(*args, **kwargs)
    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = super().__call__(*args, **kwargs)
            return self.instance
        else:
            return self.instance



class BreakingBad(metaclass=Singleton):
    pass

bestShow = BreakingBad()
secondBestShow =  BreakingBad()

print(bestShow is secondBestShow)
