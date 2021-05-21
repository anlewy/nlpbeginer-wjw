from typing import NewType, Any


class Student(object):

    def __init__(self):
        self._birth = 1999

    @property
    def birth(self):
        return self._birth

    @birth.setter
    def birth(self, value):
        self._birth = value

    @property
    def age(self):
        return 2014 - self._birth


s = Student()


def testNote():
    """
    what this function do ?

    hiahiahia
    nothing!
    """
    return s


UserId = NewType('xxx', Any)
print(UserId.__name__)


def get(x: UserId):
    return x


class c1:
    x1: int = 32
    x2: str = "abc"

    def __init__(self):
        self.x1 = 666

    @classmethod
    def print(self):
        print(self.x1)


a = c1()
a.print()
c1.print()
