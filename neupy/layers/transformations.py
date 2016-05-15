import numpy as np
import theano.tensor as T

from neupy.core.properties import ProperFractionProperty, TypedListProperty, IntProperty
from .base import BaseLayer

__all__ = ('Dropout', 'Reshape', 'Combination', 'Length2D', 'Length3D', 
    'Length4D', 'Length', 'Length2', 'Average', 'SquareMax', 'SquareNorm')

class Dropout(BaseLayer):
    """ Dropout layer

    Parameters
    ----------
    proba : float
        Fraction of the input units to drop. Value needs to be
        between 0 and 1.
    """
    proba = ProperFractionProperty(required=True)

    def __init__(self, proba, **options):
        options['proba'] = proba
        super(Dropout, self).__init__(**options)

    @property
    def size(self):
        return self.relate_to_layer.size

    def output(self, input_value):
        # Use NumPy seed to make Theano code easely reproducible
        max_possible_seed = 4e9
        seed = np.random.randint(max_possible_seed)
        theano_random = T.shared_randomstreams.RandomStreams(seed)

        proba = (1.0 - self.proba)
        mask = theano_random.binomial(n=1, p=proba,
                                      size=input_value.shape,
                                      dtype=input_value.dtype)
        return (mask * input_value) / proba

    def __repr__(self):
        return "{name}(proba={proba})".format(
            name=self.__class__.__name__,
            proba=self.proba
        )


class Reshape(BaseLayer):
    """ Gives a new shape to an input value without changing
    its data.

    Parameters
    ----------
    shape : tuple or list
        New feature shape. ``None`` value means that feature
        will be flatten in 1D vector. If you need to get the
        output feature with more that 2 dimensions then you can
        set up new feature shape using tuples. Defaults to ``None``.
    """
    shape = TypedListProperty()
    presize = IntProperty()

    def __init__(self, shape=None, presize=None, **options):
        if shape is not None:
            options['shape'] = shape
        if presize is not None:
            options['presize'] = presize
        super(Reshape, self).__init__(**options)

    def output(self, input_value):
        """ Reshape the feature space for the input value.

        Parameters
        ----------
        input_value : array-like or Theano variable
        """
        new_feature_shape = self.shape
        input_shape = input_value.shape[0]

        if new_feature_shape is None:
            output_shape = input_value.shape[1:]
            new_feature_shape = T.prod(output_shape)
            output_shape = (input_shape, new_feature_shape)
        else:
            output_shape = (input_shape,) + new_feature_shape

        return T.reshape(input_value, output_shape)
    
    def __repr__(self):
        return "{name}(shape={shape},presize={presize})".format(
            name=self.__class__.__name__,
            shape=self.shape, 
            presize=self.presize
        )

class Length2D(Reshape):
    def output(self, input_value):
        if input_value.ndim == 3:
            return (input_value[:, 0] - input_value[:, 1]).norm(2, axis = 1)
        elif input_value.ndim == 4:
            return (input_value[:, :, 0, :] - input_value[:, :, 1, :]).norm(2, axis = 2)
            # return input_value

class Length3D(Reshape):
    def output(self, input_value):
        le = [[1, -1, 0], [1, 0, -1], [0, 1, -1]]
        le = T.as_tensor_variable(np.asarray(le).T)
        if input_value.ndim == 3:
            return T.tensordot(input_value, le, axes = [1, 0]).norm(2, axis = 1)
        elif input_value.ndim == 4:
            return T.tensordot(input_value, le, axes = [2, 0]).norm(2, axis = 2)

class Length4D(Reshape):
    def output(self, input_value):
        le = [[1, -1, 0, 0], [1, 0, -1, 0], [1, 0, 0, -1], 
            [0, 1, -1, 0], [0, 1, 0, -1], [0, 0, 1, -1]]
        le = T.as_tensor_variable(np.asarray(le).T)
        if input_value.ndim == 3:
            return T.tensordot(input_value, le, axes = [1, 0]).norm(2, axis = 1)
        elif input_value.ndim == 4:
            return T.tensordot(input_value, le, axes = [2, 0]).norm(2, axis = 2)

class SquareMax(Reshape):
    def output(self, input_value):
        le = T.as_tensor_variable(np.asarray([1, -1]))
        if input_value.ndim == 3:
            return T.abs_(T.tensordot(input_value, le, axes = [1, 0])).max(axis = 1)
        elif input_value.ndim == 4:
            return T.abs_(T.tensordot(input_value, le, axes = [2, 0])).max(axis = 2)

class SquareNorm(Reshape):
    def output(self, input_value):
        le = T.as_tensor_variable(np.asarray([1, -1]))
        if input_value.ndim == 3:
            return (T.tensordot(input_value, le, axes = [1, 0])).norm(2, axis = 1)
        elif input_value.ndim == 4:
            return (T.tensordot(input_value, le, axes = [2, 0])).norm(2, axis = 2)

class Length(Reshape):
    num = IntProperty()
    def __init__(self, num=None, **options):
        if num is not None:
            options['num'] = num
        super(Length, self).__init__(**options)
    
    def output(self, input_value):
        le = np.zeros((self.num * (self.num - 1) / 2, self.num), dtype=np.int)
        k = 0
        for i in range(0, self.num):
            for j in range(0, i):
                le[k, i] = 1
                le[k, j] = -1
                k += 1
        le = T.as_tensor_variable(le.T)
        if input_value.ndim == 3:
            return T.tensordot(input_value, le, axes = [1, 0]).norm(2, axis = 1)
        elif input_value.ndim == 4:
            return T.tensordot(input_value, le, axes = [2, 0]).norm(2, axis = 2)
    
    def __repr__(self):
        return "{name}(num={num},presize={presize})".format(
            name=self.__class__.__name__,
            num=self.num, 
            presize=self.presize
        )

class Length2(Length):
    def output(self, input_value):
        return super(Length2, self).output(input_value)**2

class Average(Reshape):
    def output(self, input_value):
        if input_value.ndim == 2:
            return T.dot(input_value, T.ones((input_value.shape[1], 1))) / input_value.shape[1]
        elif input_value.ndim == 3:
            return T.dot(input_value, T.ones((input_value.shape[2], 1))) / input_value.shape[2]

class Combination(BaseLayer):
    """form all possible combination of the first dimension of the 
    input data. 
    num: num of objects to choose from.
    comb: num of objects to choose"""
    num = IntProperty()
    comb = IntProperty()

    def __init__(self, num=None, comb=None, **options):
        if num is not None:
            options['num'] = num
        if comb is not None:
            options['comb'] = comb
        super(Combination, self).__init__(**options)
        
    def output(self, input_value):
        n = self.comb
        input_shape = self.num
        import itertools
        ee = np.eye(input_shape, dtype=int)
        le = [list(g) for g in itertools.combinations(ee, n)]
        le = T.as_tensor_variable(np.asarray(le))
        return T.tensordot(le, input_value, axes = [2, 1]).swapaxes(2, 1).swapaxes(1, 0)
        