import numpy as np

class scale(object):

    """
    Makes values dimensionless:
        1. divide values by reference
        2. subtract 1 (so that dimensionless values are spread around 0 and not around 1)

    Reverts Scaling:
        1. add 1
        2. multiply by same factor as before

    """

    def __init__(self,is_already_dimensionless=False,verbose=False):

        self.is_dimless = is_already_dimensionless
        self.verbose = verbose

    def __add__(self, other):
        return np.add(self.value,other.value)

    def __mul__(self, other):
        return np.multiply(self.value,other.value)

    def __sub__(self, other):
        return np.subtract(self.value,other.value)

    def __truediv__(self, other):
        return np.divide(self.value,other.value)

    def copy(self):
        return self

    def __str__(self):
        s0 = "This is a %s\n"%self.name
        s1 = "Dimensionless factor: %.4f\n"%self.scaler
        s2 = "Values: " + str(self.value)
        return s0+s1+s2

    def __print_dim_error(self):
        if self.verbose:
            if self.is_dimless:
                s0 = "Can not make %s dimensionless as it already has no dimension."%self.name

            elif not self.is_dimless:
                s0 = "Can not invert scaling of %s as it is not dimensionless."%self.name

            print(s0)

    def T(self,temp):
        self.name = "Temperature"
        self.scaler = 273.15
        self.value = temp

        if not self.is_dimless: # only make dimensionless if not already happend so:
            self.value = np.subtract(np.divide(temp, self.scaler), 1)
            self.is_dimless = True

        else:
            self.__print_dim_error()

        return self



    def invert(self):

        if self.is_dimless: # only invert if it's invertable:
            self.value = np.multiply(np.add(self.value,1),self.scaler)
            self.is_dimless = False
        else:
            self.__print_dim_error()

        return self


if __name__ == "__main__":
    s = scale().T(298)
    print(s)
    s.invert()
    print(s.value)