from abc import ABCMeta, abstractmethod

class Show(metaclass=ABCMeta):
    @abstractmethod
    def play(self, season, episode):
        pass

    @abstractmethod
    def stop(self):
        pass


class BreakingBad(Show):
    def play(self, season, episode):
        print(f"Playing s{season}e{episode}")
    
    def stop(self):
        print("Stopping...!")

myshow = BreakingBad()
myshow.play(5, 16)
myshow.stop()