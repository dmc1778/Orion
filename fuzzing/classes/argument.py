from numpy.random import choice, randint
from enum import IntEnum
from utils.probability import *
from constants.enum import OracleType


class ArgType(IntEnum):
    INT = 1
    STR = 2
    FLOAT = 3
    BOOL = 4
    TUPLE = 5
    LIST = 6
    NULL = 7
    TORCH_OBJECT = 8
    TORCH_TENSOR = 9
    TORCH_DTYPE = 10
    TF_TENSOR = 11
    TF_DTYPE = 12
    KERAS_TENSOR = 13
    TF_VARIABLE = 14
    TF_OBJECT = 15


class Argument:
    """
    _support_types: all the types that Argument supports.
    NOTICE: The inherent class should call the method of its parent
    when it does not support its type
    """

    _support_types = [
        ArgType.INT,
        ArgType.STR,
        ArgType.FLOAT,
        ArgType.NULL,
        ArgType.TUPLE,
        ArgType.LIST,
        ArgType.BOOL,
    ]
    _int_values = [-1024, -16, -1, 0, 1, 16, 1024]
    _str_values = ["mean", "sum", "max", "zeros", "reflect", "circular", "replicate"]
    _float_values = [0.0, 1.0, -1.0, 63.0, -63.0, 1024.0, -1024.0, 1e20, -1e20]

    def __init__(self, value, type: ArgType):
        self.value = value
        self.type = type
        self.non_scalar_input_flag = False
        self.MISMATCH_OP = ''

    def activate_non_scalar_input_flag(self):
        self.non_scalar_input_flag = True

    def to_code(self, var_name: str) -> str:
        """ArgType.LIST and ArgType.TUPLE should be converted to code in the inherent class"""
        if self.type in [ArgType.INT, ArgType.FLOAT, ArgType.BOOL]:
            if self.non_scalar_input_flag:
                return f"{var_name} = tf.constant([], shape=[0], dtype=tf.float64)\n"
            else:
                return f"{var_name} = {self.value}\n"
        elif self.type == ArgType.STR:
            return f'{var_name} = "{self.value}"\n'
        elif self.type == ArgType.NULL:
            return f"{var_name} = None\n"
        else:
            assert 0

    def mutate_value(self) -> None:
        if self.type == ArgType.INT:
            self.value = self.mutate_int_value(self.value)
        elif self.type == ArgType.STR:
            self.value = self.mutate_str_value(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = self.mutate_float_value(self.value)
        elif self.type == ArgType.BOOL:
            self.value = not self.value
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            # self.value is a list now
            for arg in self.value:
                arg.mutate_value()
        elif self.type == ArgType.NULL:
            pass
        else:
            assert 0

    def mutate_type(self) -> None:
        """The type mutation for NULL should be implemented in the inherent class"""
        if self.type in [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]:
            types = [ArgType.INT, ArgType.FLOAT, ArgType.STR, ArgType.BOOL]
            types.remove(self.type)
            self.type = choice(types)
            # change the value
            if self.type == ArgType.INT:
                self.value = self.mutate_int_value(0)
            elif self.type == ArgType.FLOAT:
                self.value = self.mutate_float_value(0.0)
            elif self.type == ArgType.STR:
                self.value = self.mutate_str_value("max")
            elif self.type == ArgType.BOOL:
                self.value = choice([True, False])
        elif self.type in [ArgType.LIST, ArgType.TUPLE]:
            for arg in self.value:
                arg.mutate_type()
        else:
            # cannot change the type of assert in the general Argument
            assert 0

    def mutate_int_value(self, value, _min=None, _max=None) -> int:
        if choose_from_list():
            value = choice(Argument._int_values)
        else:
            value += randint(-64, 64)
        # min <= value <= max
        if _min != None:
            value = max(_min, value)
        if _max != None:
            value = min(_max, value)
        return value

    def mutate_str_value(self, value) -> str:
        """You can add more string mutation strategies"""
        if choose_from_list():
            return choice(Argument._str_values)
        else:
            return value

    def mutate_float_value(self, value) -> float:
        if choose_from_list():
            return choice(Argument._float_values)
        else:
            return value + randint(-64, 64) * 1.0

    def initial_value(self, type: ArgType):
        """LIST and TUPLE should be implemented in the inherent class"""
        if type == ArgType.INT:
            return choice(Argument._int_values)
        elif type == ArgType.FLOAT:
            return choice(Argument._float_values)
        elif type == ArgType.STR:
            return choice(Argument._str_values)
        elif type == ArgType.BOOL:
            return choice([True, False])
        elif type == ArgType.NULL:
            return None
        else:
            assert 0

    @staticmethod
    def get_type(x):
        if x is None:
            return ArgType.NULL
        elif isinstance(x, bool):
            return ArgType.BOOL
        elif isinstance(x, int):
            return ArgType.INT
        elif isinstance(x, str):
            return ArgType.STR
        elif isinstance(x, float):
            return ArgType.FLOAT
        elif isinstance(x, tuple):
            return ArgType.TUPLE
        elif isinstance(x, list):
            return ArgType.LIST
        else:
            return None
