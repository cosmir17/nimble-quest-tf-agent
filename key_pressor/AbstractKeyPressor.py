import abc


class AbstractKeyPressor(abc.ABC):

    @property
    @abc.abstractmethod
    def spacebar(self):
        return 'Should never reach here'

    @property
    @abc.abstractmethod
    def leftarrow(self):
        return 'Should never reach here'

    @property
    @abc.abstractmethod
    def rightarrow(self):
        return 'Should never reach here'

    @property
    @abc.abstractmethod
    def uparrow(self):
        return 'Should never reach here'

    @property
    @abc.abstractmethod
    def downarrow(self):
        return 'Should never reach here'

    @property
    @abc.abstractmethod
    def enter(self):
        return 'Should never reach here'

    @abc.abstractmethod
    def press_key(self, key):
        return


