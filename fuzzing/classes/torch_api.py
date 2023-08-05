import torch
import random
from classes.argument import *
from classes.api import *
from classes.database import TorchDatabase
from os.path import join
import numpy as np
import string


class TorchArgument(Argument):
    _supported_types = [ArgType.TORCH_DTYPE,
                        ArgType.TORCH_OBJECT, ArgType.TORCH_TENSOR]
    _dtypes = [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.complex64,
        torch.complex128,
        torch.bool,
    ]
    _memory_format = [
        torch.contiguous_format,
        torch.channels_last,
        torch.preserve_format,
    ]

    def __init__(
        self, value, type: ArgType, shape=None, dtype=None, max_value=1, min_value=0
    ):
        super().__init__(value, type)
        self.shape = shape
        self.dtype = dtype
        self.max_value = max_value
        self.min_value = min_value

        """
        Added by me
        """
        
        self.history = False
        self.llm = False
        
        self.tensor_zero_flag_type1 = False
        self.tensor_zero_flag_type2 = False
        
        self.nan_input_tensor = False
        self.nan_input_tensor_whole = False

        self.scalar_input_flag = False

        self.tensor_empty_flag_type1 = False 
        self.tensor_empty_flag_type2 = False 
        self.tensor_empty_flag_type3 = False 
        self.tensor_empty_flag_type4 = False 
        self.tensor_empty_flag_type5 = False 
        self.tensor_empty_flag_type6 = False
        self.tensor_empty_flag_type7 = False 
        
        """Large tensor
 
        """
        self.large_tensor_flag1 = False
        self.large_tensor_flag2 = False
        self.large_tensor_flag3 = False
        self.large_tensor_flag4 = False
        self.large_tensor_flag5 = False
        self.large_tensor_flag6 = False
        self.large_tensor_flag7 = False
        self.large_tensor_flag8 = False
        
        """Non scalar flags
        """
        self.non_scalar_input_flag1 = False
        self.non_scalar_input_flag2 = False
        self.non_scalar_input_flag3 = False
        self.non_scalar_input_flag4 = False
        self.non_scalar_input_flag5 = False
        
        """Large tensor
 
        """
        self.make_tensor_neg1 = False
        self.make_tensor_neg2 = False

    def to_code(self, var_name, low_precision=False, is_cuda=False) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_code(f"{var_name}_{i}",
                                              low_precision, is_cuda)
                arg_name_list += f"{var_name}_{i},"

            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code

        # elif self.type in [ArgType.INT, ArgType.FLOAT, ArgType.BOOL]:
        #     dtype = self.dtype
        #     if self.non_scalar_input_flag:
        #         code = f"{var_name}_tensor = torch.tensor([], dtype={dtype})\n"
        #     return code
        
        elif self.type == ArgType.TORCH_TENSOR:
            dtype = self.dtype
            max_value = self.max_value
            min_value = self.min_value
            if low_precision:
                dtype = self.low_precision_dtype(dtype)
                max_value, min_value = self.random_tensor_value(dtype)
            suffix = ""
            big_number = random.randint(187, 16106)
            if is_cuda:
                suffix = ".cuda()"
            if dtype.is_floating_point:
                if self.make_tensor_neg1:
                    code = f"{var_name}_tensor = torch.neg(torch.rand({self.shape}, dtype={dtype}))\n"
                elif self.make_tensor_neg2:
                    code = f"{var_name}_tensor = -torch.rand({self.shape}, dtype={dtype})\n"
                
                # non scalar 
                elif self.non_scalar_input_flag1:
                    minimum = 1
                    maximum = 100
                    length = 10  
                    random_list = [random.randint(minimum, maximum) for _ in range(length)]
                    code = f"{var_name}_tensor = torch.tensor({random_list}, dtype={dtype})\n"
                elif self.non_scalar_input_flag2:
                    start = 0
                    end = 10
                    step = 2
                    code = f"{var_name}_tensor = torch.arange({start},{end},{step})\n"
                elif self.non_scalar_input_flag3:
                    start = 0
                    end = 1
                    num_steps = 5
                    code = f"{var_name}_tensor = torch.linspace({start},{end},{num_steps})\n"
                elif self.non_scalar_input_flag4:
                    min_val = random.randint(0, 163)
                    max_val = random.randint(0, 163)
                    code = f"{var_name}_tensor = torch.zeros(({min_val},{max_val}))\n"
                elif self.non_scalar_input_flag4:
                    start = 0
                    end = 1
                    num_steps = 5
                    code = f"{var_name}_tensor = torch.linspace({start},{end},{num_steps})\n"
                    
                ### large
                elif self.large_tensor_flag1:
                    min_val = random.randint(0, 1610)
                    max_val = random.randint(0, 1610)
                    code = f"{var_name}_tensor = torch.randint({min_val},{max_val},{self.shape}, dtype={dtype})\n"
                elif self.large_tensor_flag2:
                    code = f"{var_name}_tensor = torch.tensor([{big_number}], dtype={dtype})\n"
                elif self.large_tensor_flag3:
                    min_val = random.randint(0, 1610)
                    max_val = random.randint(0, 1610)
                    code = f"{var_name}_tensor = torch.ones(({min_val}, {max_val}), dtype={dtype})\n"
                elif self.large_tensor_flag4:
                    min_val = random.randint(0, 1610)
                    max_val = random.randint(0, 1610)
                    code = f"{var_name}_tensor = torch.rand(({min_val},{max_val}), dtype={dtype})\n"
                elif self.large_tensor_flag5:
                    min_val = random.randint(0, 1610)
                    max_val = random.randint(0, 1610)
                    code = f"{var_name}_tensor = torch.randn(({min_val},{max_val}), dtype={dtype})\n"
                elif self.large_tensor_flag6:
                    min_val = random.randint(0, 1000)
                    max_val = random.randint(0, 1000)
                    random_indx = random.randint(10,100)
                    code = f"{var_name}_tensor = torch.full(({min_val},{max_val}),random_indx, dtype={dtype})\n"
                elif self.large_tensor_flag7:
                    min_val = random.randint(0, 1000)
                    max_val = random.randint(0, 1000)
                    code = f"{var_name}_tensor = torch.arrange({min_val},{max_val}, dtype={dtype})\n"
                elif self.large_tensor_flag8:
                    code = f"{var_name}_tensor = torch.linspace(0,1, 1000000, dtype={dtype})\n"
                
                    
                elif self.tensor_empty_flag_type1:
                    code = f"{var_name}_tensor = torch.tensor([])\n"
                elif self.tensor_empty_flag_type2:
                    code = f"{var_name}_tensor = torch.full([0], {big_number}, dtype={dtype})\n"
                elif self.tensor_empty_flag_type3:
                    code = f"{var_name}_tensor = torch.ones((0,0))\n"
                elif self.tensor_empty_flag_type4:
                    min_val = random.randint(0, 1000)
                    max_val = random.randint(0, 1000)
                    code = f"{var_name}_tensor = torch.zeros(({min_val},{max_val}))\n"
                elif self.tensor_empty_flag_type5:
                    code = f"{var_name}_tensor = torch.tensor([])\n"
                elif self.tensor_empty_flag_type6:
                    code = f"{var_name}_tensor = torch.tensor([])\n"
                elif self.tensor_empty_flag_type7:
                    code = f"{var_name}_tensor = torch.tensor([])\n"
                    
                    
                elif self.tensor_zero_flag_type1:
                    code = f"{var_name}_tensor = torch.zeros({self.shape}, dtype={dtype})\n"
                elif self.tensor_zero_flag_type2:
                    val1 = random.randint(100, 1000)
                    val2 = random.randint(100, 1000)
                    val3 = random.randint(100, 1000)
                    val4 = random.randint(100, 1000)
                    val5 = random.randint(100, 1000)
                    code = f"{var_name}_tensor = torch.zeros([{val1}, {val2}, {val3}, {val4}, {val5}], dtype={dtype})\n"
                elif self.nan_input_tensor:
                    code = f"{var_name}_tensor = torch.tensor({self.shape}, dtype={dtype})\n"
                    # code += f"{var_name}_tensor[{var_name}_tensor == {self.shape[0]}] = float('nan')\n"
                elif self.nan_input_tensor_whole:
                    code = f"{var_name}_tensor = np.nan \n"
                elif self.scalar_input_flag:
                    x = random.randint(187, 1610)
                    code = f"{var_name}_tensor = {x} \n"
                else:
                    code = (
                        f"{var_name}_tensor = torch.rand({self.shape}, dtype={dtype})\n"
                    )
            elif dtype.is_complex:
                if self.make_tensor_neg1:
                    code = f"{var_name}_tensor = torch.neg(torch.rand({self.shape}, dtype={dtype}))\n"
                elif self.make_tensor_neg2:
                       code = f"{var_name}_tensor = -torch.rand({self.shape}, dtype={dtype})\n" 
                       
                # non scalar 
                elif self.non_scalar_input_flag1:
                    minimum = 1
                    maximum = 100
                    length = 10  
                    random_list = [random.randint(minimum, maximum) for _ in range(length)]
                    code = f"{var_name}_tensor = torch.tensor({random_list}, dtype=torch.complex128)\n"
                elif self.non_scalar_input_flag2:
                    start = 0
                    end = 10
                    step = 2
                    code = f"{var_name}_tensor = torch.arange({start},{end},{step}, dtype=torch.complex128)\n"
                elif self.non_scalar_input_flag3:
                    start = 0
                    end = 1
                    num_steps = 5
                    code = f"{var_name}_tensor = torch.linspace({start},{end},{num_steps}, dtype=torch.complex128)\n"
                elif self.non_scalar_input_flag4:
                    min_val = random.randint(0, 163)
                    max_val = random.randint(0, 163)
                    code = f"{var_name}_tensor = torch.zeros(({min_val},{max_val}), dtype=torch.complex128)\n"
                elif self.non_scalar_input_flag4:
                    start = 0
                    end = 1
                    num_steps = 5
                    code = f"{var_name}_tensor = torch.linspace({start},{end},{num_steps}, dtype=torch.complex128)\n"
                    
                ### large
                elif self.large_tensor_flag1:
                    min_val = random.randint(0, 1610)
                    max_val = random.randint(0, 1610)
                    code = f"{var_name}_tensor = torch.randint({min_val},{max_val},{self.shape}, dtype={dtype})\n"
                elif self.large_tensor_flag2:
                    code = f"{var_name}_tensor = torch.tensor([{big_number}], dtype={dtype})\n"
                elif self.large_tensor_flag3:
                    min_val = random.randint(0, 1610)
                    max_val = random.randint(0, 1610)
                    code = f"{var_name}_tensor = torch.ones(({min_val}, {max_val}), dtype={dtype})\n"
                elif self.large_tensor_flag4:
                    min_val = random.randint(0, 1610)
                    max_val = random.randint(0, 1610)
                    code = f"{var_name}_tensor = torch.rand(({min_val},{max_val}), dtype={dtype})\n"
                elif self.large_tensor_flag5:
                    min_val = random.randint(0, 1610)
                    max_val = random.randint(0, 1610)
                    code = f"{var_name}_tensor = torch.randn(({min_val},{max_val}), dtype={dtype})\n"
                elif self.large_tensor_flag6:
                    min_val = random.randint(0, 1000)
                    max_val = random.randint(0, 1000)
                    random_indx = random.randint(10,100)
                    code = f"{var_name}_tensor = torch.full(({min_val},{max_val}),random_indx, dtype={dtype})\n"
                elif self.large_tensor_flag7:
                    min_val = random.randint(0, 1000000)
                    max_val = random.randint(0, 1000000)
                    code = f"{var_name}_tensor = torch.arrange({min_val},{max_val}, dtype={dtype})\n"
                elif self.large_tensor_flag2:
                    code = f"{var_name}_tensor = torch.linspace(0,1, 1000000, dtype={dtype})\n"
                    
                elif self.tensor_empty_flag_type1:
                    code = f"{var_name}_tensor = torch.tensor([])\n"
                elif self.tensor_empty_flag_type2:
                    code = f"{var_name}_tensor = torch.full([0], {big_number}, dtype={dtype})\n"
                elif self.tensor_empty_flag_type3:
                    code = f"{var_name}_tensor = torch.ones((0,0))\n"
                elif self.tensor_empty_flag_type4:
                    min_val = random.randint(0, 1000000)
                    max_val = random.randint(0, 1000000)
                    code = f"{var_name}_tensor = torch.zeros(({min_val},{max_val}))\n"
                elif self.tensor_empty_flag_type5:
                    code = f"{var_name}_tensor = torch.tensor([])\n"
                elif self.tensor_empty_flag_type6:
                    code = f"{var_name}_tensor = torch.tensor([])\n"
                elif self.tensor_empty_flag_type7:
                    code = f"{var_name}_tensor = torch.tensor([])\n"
                    
                elif self.tensor_empty_flag_type1:
                    code = f"{var_name}_tensor = torch.tensor([], dtype={dtype})\n"
                elif self.tensor_empty_flag_type2:
                    code = f"{var_name}_tensor = torch.full([0], {big_number}, dtype={dtype})\n"
                elif self.tensor_zero_flag_type1:
                    code = f"{var_name}_tensor = torch.zeros({self.shape}, dtype={dtype})\n"
                elif self.tensor_zero_flag_type2:
                    val1 = random.randint(100, 1000)
                    val2 = random.randint(100, 1000)
                    val3 = random.randint(100, 1000)
                    val4 = random.randint(100, 1000)
                    val5 = random.randint(100, 1000)
                    code = f"{var_name}_tensor = torch.zeros([{val1}, {val2}, {val3}, {val4}, {val5}], dtype={dtype})\n"
                elif self.nan_input_tensor:
                    code = f"{var_name}_tensor = torch.tensor({self.shape}, dtype={dtype})\n"
                    # code += f"{var_name}_tensor[{var_name}_tensor == {self.shape}] = float('nan')\n"
                elif self.nan_input_tensor_whole:
                    code = f"{var_name}_tensor = np.nan \n"
                elif self.scalar_input_flag:
                    x = random.randint(187, 1610)
                    code = f"{var_name}_tensor = {x} \n"
                else:
                    code = (
                        f"{var_name}_tensor = torch.rand({self.shape}, dtype={dtype})\n"
                    )
            elif dtype == torch.bool:
                
                if self.large_tensor_flag1:
                    big_val = random.randint(161, 1000)
                    code = f"{var_name}_tensor = torch.full((1, 1, 1, 1, 1,), {big_val}, dtype={dtype})\n"
                elif self.large_tensor_flag2:
                    min_val = random.randint(187, 1610)
                    max_val = random.randint(187, 1610)
                    code = f"{var_name}_tensor = torch.randint({min_val},{max_val},{self.shape}, dtype={dtype})\n"
                elif self.large_tensor_flag3:
                    min_val = random.randint(187, 1610)
                    max_val = random.randint(187, 1610)
                    code = f"{var_name}_tensor = torch.ones(({min_val},{max_val}), dtype={dtype})\n"
                elif self.large_tensor_flag4:
                    min_val = random.randint(187, 1610)
                    max_val = random.randint(187, 1610)
                    code = f"{var_name}_tensor = torch.empty(({min_val},{max_val}), dtype={dtype})\n"
                elif self.large_tensor_flag5:
                    min_val = random.randint(187, 1610)
                    max_val = random.randint(187, 1610)
                    code = f"{var_name}_tensor = torch.full(({min_val},{max_val}),True, dtype={dtype})\n"
                elif self.large_tensor_flag6:
                    min_val = random.randint(187, 1610)
                    max_val = random.randint(187, 1610)
                    code = f"{var_name}_tensor = torch.eye({min_val}, dtype={dtype})\n"
    
                elif self.tensor_empty_flag_type1:
                    code = f"{var_name}_tensor = torch.tensor([], dtype={dtype})\n"
                elif self.tensor_empty_flag_type2:
                    code = f"{var_name}_tensor = torch.full((0, 0), False)\n"
                elif self.tensor_empty_flag_type3:
                    code = f"{var_name}_tensor = torch.ones((0,0), dtype={dtype})\n"
                elif self.tensor_empty_flag_type4:
                    code = f"{var_name}_tensor = torch.ones((0,0), dtype={dtype})\n"
                elif self.tensor_empty_flag_type5:
                    code = f"{var_name}_tensor = torch.tensor([], dtype={dtype})\n"
                elif self.tensor_empty_flag_type6:
                    code = f"{var_name}_tensor = torch.tensor([], dtype={dtype})\n"
                elif self.tensor_empty_flag_type7:
                    code = f"{var_name}_tensor = torch.tensor([], dtype={dtype})\n"
                    
                elif self.tensor_zero_flag_type1:
                    code = f"{var_name}_tensor = torch.zeros({self.shape}, dtype={dtype})\n"
                elif self.tensor_zero_flag_type2:
                    val1 = random.randint(100, 1000)
                    val2 = random.randint(100, 1000)
                    val3 = random.randint(100, 1000)
                    val4 = random.randint(100, 1000)
                    val5 = random.randint(100, 1000)
                    code = f"{var_name}_tensor = torch.zeros([{val1}, {val2}, {val3}, {val4}, {val5}], dtype={dtype})\n"
                elif self.nan_input_tensor:
                    code = f"{var_name}_tensor = torch.tensor({self.shape}, dtype={dtype})\n"
                    # code += f"{var_name}_tensor[{var_name}_tensor == {self.shape[0]}] = float('nan')\n"
                elif self.nan_input_tensor_whole:
                    code = f"{var_name}_tensor = np.nan \n"
                elif self.scalar_input_flag:
                    x = random.randint(1879048192, 161063793887434)
                    code = f"{var_name}_tensor = {x} \n"
                else:
                    code = f"{var_name}_tensor = torch.randint(0,2,{self.shape}, dtype={dtype})\n"
            else:
                code = f"{var_name}_tensor = torch.randint({min_value},{max_value},{self.shape}, dtype={dtype})\n"
            code += f"{var_name} = {var_name}_tensor.clone(){suffix}\n"
            return code
        elif self.type == ArgType.TORCH_OBJECT:
            return f"{var_name} = {self.value}\n"
        elif self.type == ArgType.TORCH_DTYPE:
            return f"{var_name} = {self.value}\n"
        return super().to_code(var_name)

    def to_diff_code(self, var_name, oracle):
        """differential testing with oracle"""
        code = ""
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_diff_code(f"{var_name}_{i}", oracle)
                arg_name_list += f"{var_name}_{i},"
            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
        elif self.type == ArgType.TORCH_TENSOR:
            if oracle == OracleType.CUDA:
                code += f"{var_name} = {var_name}_tensor.clone().cuda()\n"
            elif oracle == OracleType.PRECISION:
                code += f"{var_name} = {var_name}_tensor.clone().type({self.dtype})\n"
        return code

    def mutate_value(self) -> None:
        if self.type == ArgType.TORCH_OBJECT:
            pass
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TORCH_TENSOR:
            self.max_value, self.min_value = self.random_tensor_value(
                self.dtype)
        elif self.type in super()._support_types:
            super().mutate_value()
        else:
            print(self.type, self.value)
            assert 0

    def mutate_type(self) -> None:
        if self.type == ArgType.NULL:
            # choose from all types
            new_type = choice(self._support_types + super()._support_types)
            self.type = new_type
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:
                self.value = [
                    TorchArgument(2, ArgType.INT),
                    TorchArgument(3, ArgType.INT),
                ]
            elif new_type == ArgType.TORCH_TENSOR:
                self.shape = [2, 2]
                self.dtype = torch.float32
            elif new_type == ArgType.TORCH_DTYPE:
                self.value = choice(self._dtypes)
            elif new_type == ArgType.TORCH_OBJECT:
                self.value = choice(self._memory_format)
            else:
                self.value = super().initial_value(new_type)
        elif self.type == ArgType.TORCH_TENSOR:
            self.dtype = choice(self._dtypes)
            self.max_value, self.min_value = self.random_tensor_value(
                self.dtype)
        elif self.type == ArgType.TORCH_OBJECT:
            pass
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type in super()._support_types:
            super().mutate_type()
        else:
            print(self.type, self.value)
            assert 0

    @staticmethod
    def random_tensor_value(dtype):
        max_value = 1
        min_value = 0
        if dtype == torch.bool:
            max_value = 2
            min_value = 0
        elif dtype == torch.uint8:
            max_value = 1 << randint(0, 9)
            min_value = 0
        elif dtype == torch.int8:
            max_value = 1 << randint(0, 8)
            min_value = -1 << randint(0, 8)
        elif dtype == torch.int16:
            max_value = 1 << randint(0, 16)
            min_value = -1 << randint(0, 16)
        else:
            max_value = 1 << randint(0, 16)
            min_value = -1 << randint(0, 16)
        return max_value, min_value

    @staticmethod
    def generate_arg_from_signature(signature):
        """Generate a Torch argument from the signature"""
        if signature == "torchTensor":
            return TorchArgument(
                None, ArgType.TORCH_TENSOR, shape=[2, 2], dtype=torch.float32
            )
        if signature == "torchdtype":
            return TorchArgument(choice(TorchArgument._dtypes), ArgType.TORCH_DTYPE)
        if isinstance(signature, str) and signature == "torchdevice":
            value = torch.device("cpu")
            return TorchArgument(value, ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature == "torchmemory_format":
            value = choice(TorchArgument._memory_format)
            return TorchArgument(value, ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature == "torch.strided":
            return TorchArgument("torch.strided", ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature.startswith("torch."):
            value = eval(signature)
            if isinstance(value, torch.dtype):
                return TorchArgument(value, ArgType.TORCH_DTYPE)
            elif isinstance(value, torch.memory_format):
                return TorchArgument(value, ArgType.TORCH_OBJECT)
            print(signature)
            assert 0
        if isinstance(signature, bool):
            return TorchArgument(signature, ArgType.BOOL)
        if isinstance(signature, int):
            return TorchArgument(signature, ArgType.INT)
        if isinstance(signature, str):
            return TorchArgument(signature, ArgType.STR)
        if isinstance(signature, float):
            return TorchArgument(signature, ArgType.FLOAT)
        if isinstance(signature, tuple):
            value = []
            for elem in signature:
                value.append(TorchArgument.generate_arg_from_signature(elem))
            return TorchArgument(value, ArgType.TUPLE)
        if isinstance(signature, list):
            value = []
            for elem in signature:
                value.append(TorchArgument.generate_arg_from_signature(elem))
            return TorchArgument(value, ArgType.LIST)
        # signature is a dictionary
        if isinstance(signature, dict):
            if not ("shape" in signature.keys() and "dtype" in signature.keys()):
                raise Exception("Wrong signature {0}".format(signature))
            shape = signature["shape"]
            dtype = signature["dtype"]
            # signature is a ndarray or tensor.
            if isinstance(shape, (list, tuple)):
                if not dtype.startswith("torch."):
                    dtype = f"torch.{dtype}"
                dtype = eval(dtype)
                max_value, min_value = TorchArgument.random_tensor_value(dtype)
                return TorchArgument(
                    None,
                    ArgType.TORCH_TENSOR,
                    shape,
                    dtype=dtype,
                    max_value=max_value,
                    min_value=min_value,
                )
            else:
                return TorchArgument(
                    None, ArgType.TORCH_TENSOR, shape=[2, 2], dtype=torch.float32
                )
        return TorchArgument(None, ArgType.NULL)

    @staticmethod
    def low_precision_dtype(dtype):
        if dtype in [torch.int16, torch.int32, torch.int64]:
            return torch.int8
        elif dtype in [torch.float32, torch.float64]:
            return torch.float16
        elif dtype in [torch.complex64, torch.complex128]:
            return torch.complex32
        return dtype

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res != None:
            return res
        if isinstance(x, torch.Tensor):
            return ArgType.TORCH_TENSOR
        elif isinstance(x, torch.dtype):
            return ArgType.TORCH_DTYPE
        else:
            return ArgType.TORCH_OBJECT

    """
    ########################################################################
    """

    def make_int_negative(self, value) -> int:
        value = random.randint(1, 100)
        new_value = -value
        return new_value

    def make_float_negative(self, value) -> float:
        value = random.uniform(0, 1)
        new_value = -value
        return new_value

    def make_tensor_negative(self) -> None:
        self.make_tensor_neg = True

    def make_bool_inverse(self, value) -> bool:
        return not value

    def mutate_negative(self) -> None:
        if self.type == ArgType.INT:
            self.value = self.make_int_negative(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = self.make_float_negative(self.value)
        elif self.type == ArgType.BOOL:
            self.value = self.make_bool_inverse(self.value)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for self in self.value:
                self.mutate_negative()
        elif self.type == ArgType.TORCH_TENSOR:
            self.make_tensor_negative()
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TORCH_OBJECT:
            self.value = None
            pass
        elif self.type == ArgType.NULL:
            self.value = None
            pass
        else:
            return

    def alter_tensor_shape(self, old_shape, reduction=True):
        new_shape = old_shape
        # Change rank
        max_n = random.randint(1, 3)
        if reduction:
            new_shape.pop()
        else:
            for i in range(max_n):
                new_shape.append(max_n)

        return new_shape

    def modify_rank(self) -> None:
        if self.type == ArgType.TORCH_TENSOR:
            if len(self.shape) == 4:
                self.shape = self.alter_tensor_shape(self.shape)
            elif len(self.shape) == 3:
                self.shape = self.alter_tensor_shape(self.shape)
            elif len(self.shape) == 2:
                self.shape = self.alter_tensor_shape(self.shape)
            elif len(self.shape) == 1:
                self.shape = self.alter_tensor_shape(self.shape)
            elif len(self.shape) == 0:
                self.shape = self.alter_tensor_shape(
                    self.shape, reduction=False)
            else:
                self.shape = self.alter_tensor_shape(self.shape)
        else:
            return

    def make_tensor_empty_type1(self) -> None:
        if self.type == ArgType.TORCH_TENSOR:
            self.tensor_empty_flag_type1 = True

    def make_tensor_empty_type2(self) -> None:
        if self.type == ArgType.TORCH_TENSOR:
            self.tensor_empty_flag_type2 = True

    def make_int_zero(self, value) -> int:
        new_value = 0
        return new_value

    def make_float_zero(self, value) -> float:
        new_value = 0
        return new_value

    def make_list_tuple_empty(self):
        if self.type == ArgType.INT:
            self.value = self.make_int_zero(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = self.make_float_zero(self.value)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for self in self.value:
                self.make_list_tuple_empty()
        else:
            return

    def make_tensor_zero_type1(self) -> None:
        if self.type == ArgType.TORCH_TENSOR:
            self.tensor_zero_flag_type1 = True

    def make_tensor_zero_type2(self) -> None:
        if self.type == ArgType.TORCH_TENSOR:
            self.tensor_zero_flag_type2 = True

    def make_tensor_nan(self) -> None:
        if self.type == ArgType.TORCH_TENSOR:
            self.nan_input_tensor = True

    def make_tensor_nan_whole(self) -> None:
        if self.type == ArgType.TORCH_TENSOR:
            self.nan_input_tensor_whole = True

    def make_input_non_scalar(self):
        if self.type == ArgType.INT:
            self.non_scalar_input_flag = True
        elif self.type == ArgType.FLOAT:
            self.non_scalar_input_flag = True
        elif self.type == ArgType.BOOL:
            self.non_scalar_input_flag = True
        else:
            return

    def make_tensor_large_type1(self) -> None:
        if self.type == ArgType.TORCH_TENSOR:
            self.large_tensor_flag1 = True

    def make_tensor_large_type2(self) -> None:
        if self.type == ArgType.TORCH_TENSOR:
            self.large_tensor_flag2 = True

    def make_input_scalar(self):
        self.scalar_input_flag = True

    def make_list_element_large(self):
        if self.type == ArgType.INT:
            self.value = random.randint(125, 1250)
        elif self.type == ArgType.FLOAT:
            self.value = random.uniform(0, 1)
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for self in self.value:
                self.make_list_element_large()
        else:
            return

    def mutate_preemptives(self) -> None:
        if self.type == ArgType.INT:
            self.value = np.nan
        elif self.type == ArgType.FLOAT:
            self.value = np.nan
        elif self.type == ArgType.STR:
            self.value = np.nan
        elif self.type == ArgType.BOOL:
            self.value = np.nan
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for self in self.value:
                self.mutate_preemptives()
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TORCH_OBJECT:
            self.value = None
            pass
        elif self.type == ArgType.NULL:
            self.value = None
            pass
        else:
            return

    def new_mutation_multiple(self, RULE=None):
        if RULE == "MUTATE_PREEMPTIVES":
            self.mutate_preemptives()
        elif RULE == "NEGATE_INT_TENSOR":
            self.mutate_negative()
        elif RULE == "RANK_REDUCTION_EXPANSION":
            self.modify_rank()
        elif RULE == "EMPTY_TENSOR_TYPE1":
            self.make_tensor_empty_type1()
        elif RULE == "EMPTY_TENSOR_TYPE2":
            self.make_tensor_empty_type2()
        elif RULE == "EMPTY_LIST":
            self.make_list_tuple_empty()
        elif RULE == "LARGE_TENSOR_TYPE1":
            self.make_tensor_large_type1()
        elif RULE == "LARGE_TENSOR_TYPE2":
            self.make_tensor_large_type2()
        elif RULE == "LARGE_LIST_ELEMENT":
            self.make_list_element_large()
        elif RULE == "ZERO_TENSOR_TYPE1":
            self.make_tensor_zero_type1()
        elif RULE == "ZERO_TENSOR_TYPE2":
            self.make_tensor_zero_type2()
        elif RULE == "NAN_TENSOR":
            self.make_tensor_nan()
        elif RULE == "NAN_TENSOR_WHOLE":
            self.make_tensor_nan_whole()
        elif RULE == "NON_SCALAR_INPUT":
            self.make_input_non_scalar()
        elif RULE == "SCALAR_INPUT":
            self.make_input_scalar()
        else:
            return

    def increase_integer(self, value) -> int:
        new_value = random.randint(1, 1000)
        val = -new_value
        return new_value

    def new_mutation(self, RULE=None):
        if self.type == ArgType.INT:
            self.value = self.increase_integer(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = self.make_float_negative(self.value)
        elif self.type == ArgType.BOOL:
            self.value = self.make_bool_inverse(self.value)
        elif self.type == ArgType.STR:
            self.value = np.nan
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for self in self.value:
                self.new_mutation()
        elif self.type == ArgType.TORCH_TENSOR:
            self.modify_rank()
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TORCH_OBJECT:
            self.value = None
            pass
        elif self.type == ArgType.NULL:
            self.value = None
            pass
        else:
            return


################################################ NEW implementation ########################################
    def mutate_bool(self,random_=False, none=False, nan=False, empty=False, zero=False):
        if nan:
            self.value = float('nan')
        elif empty:
            self.value = [()]
        elif zero:
            self.value = 0.0
        elif none:
            self.value = None
        elif random_:
            b_ = random.choice([True, False])
            self.value = b_
        else:
            return
    
   
    def mutate_integer(self, zero=False, large=False, neg=False, neg_large=False, nan=False, none=False, empty=False) -> int:
        if zero:
            self.value = 0
        elif large:
            new_value = random.choice([1250999896764, 1250999896765, 1250999896766, 1250999900000])
            self.value = new_value
        elif neg:
            new_value = random.randint(1e3, 1e5)
            self.value = -new_value
        elif neg_large:
            negs = [125,
            100,
            9999,
            1234,
            100,
            100,
            1000,
            100,
            10,
            1,
            1000,
            1234,
            9876,
            1234,
            1000,
            9999,
            1000,
            123,
            9876,
            1000,
            9999,
            1000,
            123210,
            98765,
            10000,
            99999]
            new_value = random.choice(negs)
            self.value = -new_value
        elif nan:
            self.value = float('nan')
        elif none:
            self.value = None
        elif empty:
            self.value = [()]
        else:
            new_value = random.randint(1e3, 1e5)
            val_ = -new_value
            self.value = val_
   
    def mutate_float(self, zero=False, large=False, neg=False, neg_large=False, nan=False, none=False, empty=False) -> float:
        if zero:
            self.value = 0.0
        elif large:
            new_value = random.choice([3.402823e+38, 1.986e+67])
            self.value = new_value
        elif neg_large:
            negs_large = [-1.0,
            -100.0,
            -1000.0,
            -10000.0,
            -100000.0,
            -1000000.0,
            -10000000.0,
            -100000000.0,
            -1000000000.0]
            self.value = random.choice(negs_large)   
        elif neg:
            negs = [-1.9,
            -1.8,
            -1.7,
            -1.6,
            -1.5,
            -1.4,
            -1.3,
            -1.2,
            -1.1,
            -1.0]
            self.value = random.choice(negs)
        elif nan:
            self.value = float('nan')
        elif none:
            self.value = None
        elif empty:
            self.value = [[]]
        else:
            new_value = random.random()
            val_ = -new_value
            self.value = val_

    def generate_junk_string(self, length):
        characters = string.ascii_letters + string.digits + string.punctuation
        junk_string = ''.join(random.choice(characters) for _ in range(length))
        return junk_string
    
    def mutate_str(self, invalid=False, empty1=False, nan=False, none=False, empty2=False) -> str:
        if invalid:
            # garbage_bytes = bytes([random.randint(0, 255)
            #                       for _ in range(len(self.value))])
            # garbage_string = repr(garbage_bytes)
            #self.value = self.generate_junk_string(len(self.value))
            
            str_vals = [
            "../../../../../../etc/passwd",
            "DROP TABLE users; --",
            "<script>alert('XSS');</script>",
            "SELECT * FROM users WHERE username='admin' OR '1'='1'",
            "<?php echo shell_exec('cat /etc/passwd');?>",
            "rm -rf /",
            "1'; DROP TABLE users; --",
            " OR 1=1; DROP TABLE users; --",
            "<script>alert(document.cookie)</script>",
            "<?php system($_GET['cmd']); ?>",
            "../../../../../../etc/passwd",
            "OR '1'='1'; --",
            "<img src='x' onerror='alert(document.cookie)'>"]
            self.value = "8.8"
        if nan:
            self.value = float('nan')
        elif none:
            self.value = None
        elif empty1:
            self.value = []
        elif empty2:
            self.value = "8.8"

 
    def modify_tensor_rank(self, large=False, neg=False, zero=False, empty=False, neg_large=False):
        if large:
            if isinstance(self.shape, list) or isinstance(self.shape, tuple):
                for i in range(len(self.shape)):
                    self.shape[i] = random.randint(20, 50)
            if isinstance(self.shape, int):
                self.mutate_integer(self.shape, large=True)
        elif neg:
            if isinstance(self.shape, list) or isinstance(self.shape, tuple):
                for i in range(len(self.shape)):
                    new_val = random.randint(1, 10)
                    self.shape[i] = -new_val
            if isinstance(self.shape, int):
                self.mutate_integer(self.shape, neg_large=True)
        elif zero:
            if isinstance(self.shape, list) or isinstance(self.shape, tuple):
                for i in range(len(self.shape)):
                    self.shape[i] = 0
            if isinstance(self.shape, int):
                self.mutate_integer(self.shape, zero=True)
        elif empty:
            if isinstance(self.shape, list) or isinstance(self.shape, tuple):
                self.shape = []
            if isinstance(self.shape, int):
                self.mutate_integer(self.shape, empty=True)
        else:
            return

    def each_arg_mutation(self, partition):
        # TENSOR
        if partition == 'NULL':
            return
        if partition == 'NULL_TF_OBJ':
            return
        if partition == 'NON_SCALAR_INPUT_TENSOR':
            if self.type == ArgType.TORCH_TENSOR:
                super().activate_non_scalar_input_flag()
            else:
                return
            
        if partition == 'NON_SCALAR_INPUT_TENSOR':
            ops = [
                lambda: setattr(self, 'non_scalar_input_flag1', True),
                lambda: setattr(self, 'non_scalar_input_flag2', True),
                lambda: setattr(self, 'non_scalar_input_flag3', True),
                lambda: setattr(self, 'non_scalar_input_flag4', True),
                lambda: setattr(self, 'non_scalar_input_flag5', True)
            ]

            selected_op = random.choice(ops)
            selected_op()
            
        if partition == 'LARGE_INPUT_TENSOR':
            ops = [
                lambda: setattr(self, 'large_tensor_flag1', True),
                lambda: setattr(self, 'large_tensor_flag2', True),
                lambda: setattr(self, 'large_tensor_flag3', True),
                lambda: setattr(self, 'large_tensor_flag4', True),
                lambda: setattr(self, 'large_tensor_flag5', True),
                lambda: setattr(self, 'large_tensor_flag6', True),
                lambda: setattr(self, 'large_tensor_flag7', True),
            ]
        
            selected_op = random.choice(ops)
            selected_op()
        if partition == 'NEGATIVE_INPUT_TENSOR':
            ops = [
                lambda: setattr(self, 'make_tensor_neg1', True),
                lambda: setattr(self, 'make_tensor_neg2', True),
            ]
        
            selected_op = random.choice(ops)
            selected_op()
        if partition == 'SCALAR_INPUT_TENSOR':
            self.scalar_input_flag = True
        if partition == 'NAN_INPUT_TENSOR':
            self.nan_input_tensor = True
        if partition == 'NAN_INPUT_TENSOR_WHOLE':
            self.nan_input_tensor_whole = True
        if partition == 'TENSOR_EMPTY_FLAG':
            ops = [
                lambda: setattr(self, 'tensor_empty_flag_type1', True),
                lambda: setattr(self, 'tensor_empty_flag_type2', True),
                lambda: setattr(self, 'tensor_empty_flag_type3', True),
                lambda: setattr(self, 'tensor_empty_flag_type4', True),
                lambda: setattr(self, 'tensor_empty_flag_type5', True),
                lambda: setattr(self, 'tensor_empty_flag_type6', True),
                lambda: setattr(self, 'tensor_empty_flag_type7', True),
            ]
        
            selected_op = random.choice(ops)
            selected_op()

        # LIST
        if partition == 'LARGE_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TORCH_TENSOR:
                    self.modify_tensor_rank(large=True)
                elif self.type == ArgType.INT:
                    self.mutate_integer(large=True)
                elif self.type == ArgType.FLOAT:
                    self.mutate_float(large=True)
                else:
                    return
        if partition == 'NEGATIVE_LARGE_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TORCH_TENSOR:
                    self.modify_tensor_rank(neg_large=True)
                elif self.type == ArgType.INT:
                    self.mutate_integer(neg_large=True)
                elif self.type == ArgType.FLOAT:
                    self.mutate_float(neg_large=True)
                else:
                    return
        if partition == 'ZERO_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TORCH_TENSOR:
                    self.modify_tensor_rank(zero=True)
                elif self.type == ArgType.INT:
                    self.mutate_integer(zero=True)
                elif self.type == ArgType.FLOAT:
                    self.mutate_float(zero=True)
                else:
                    return
        if partition == 'NEGATIVE_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TORCH_TENSOR:
                    self.modify_tensor_rank(neg=True)
                elif self.type == ArgType.INT:
                    self.mutate_integer(neg=True)
                elif self.type == ArgType.FLOAT:
                    self.mutate_float(neg=True)
                else:
                    return
        if partition == 'EMPTY_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TORCH_TENSOR:
                    self.value = ['']
                elif self.type == ArgType.INT:
                    self.value = ['']
                elif self.type == ArgType.FLOAT:
                    self.value = ['']
                elif self.type == ArgType.STR:
                    self.value = ['']
                else:
                    return
        if partition == 'INVALID_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.STR:
                    self.mutate_str(invalid=True)
        if partition == 'NONE_INPUT_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TORCH_TENSOR:
                    self.value = None
                elif self.type == ArgType.INT:
                    self.value = None
                elif self.type == ArgType.FLOAT:
                    self.value = None
                elif self.type == ArgType.STR:
                    self.value = None
                else:
                    return
        if partition == 'NAN_INPUT_LIST_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TORCH_TENSOR:
                    self.value = float('nan')
                elif self.type == ArgType.INT:
                    self.value = float('nan')
                elif self.type == ArgType.FLOAT:
                    self.value = float('nan')
                elif self.type == ArgType.STR:
                    self.value = float('nan')
                else:
                    return
        # TUPLE
        if partition == 'LARGE_TUPLE_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TORCH_TENSOR:
                    self.modify_tensor_rank(large=True)
                elif self.type == ArgType.INT:
                    self.mutate_integer(large=True)
                elif self.type == ArgType.FLOAT:
                    self.mutate_float(large=True)
                else:
                    return
        if partition == 'ZERO_TUPLE_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TORCH_TENSOR:
                    self.modify_tensor_rank(zero=True)
                elif self.type == ArgType.INT:
                    self.mutate_integer(zero=True)
                elif self.type == ArgType.FLOAT:
                    self.mutate_float(zero=True)
                else:
                    return
        if partition == 'NEGATIVE_TUPLE_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TORCH_TENSOR:
                    self.modify_tensor_rank(neg=True)
                elif self.type == ArgType.INT:
                    self.mutate_integer(neg=True)
                elif self.type == ArgType.FLOAT:
                    self.mutate_float(neg=True)
                else:
                    return
        if partition == 'EMPTY_TUPLE_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TORCH_TENSOR:
                    self.value = ['']
                elif self.type == ArgType.INT:
                    self.value = ['']
                elif self.type == ArgType.FLOAT:
                    self.value = ['']
                elif self.type == ArgType.STR:
                    self.value = ['']
                else:
                    return
        if partition == 'INVALID_TUPLE_ELEMENT':
            for self in self.value:
                if self.type == ArgType.STR:
                    self.mutate_str(invalid=True)
        if partition == 'NONE_INPUT_TUPLE_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TORCH_TENSOR:
                    self.value = None
                elif self.type == ArgType.INT:
                    self.value = None
                elif self.type == ArgType.FLOAT:
                    self.value = None
                elif self.type == ArgType.STR:
                    self.value = None
                else:
                    return
        if partition == 'NAN_INPUT_TUPLE_ELEMENT':
            for self in self.value:
                if self.type == ArgType.TORCH_TENSOR:
                    self.value = float('nan')
                elif self.type == ArgType.INT:
                    self.value = float('nan')
                elif self.type == ArgType.FLOAT:
                    self.value = float('nan')
                elif self.type == ArgType.STR:
                    self.value = float('nan')
                else:
                    return

        # INT
        if partition == 'NEGATIVE_INTEGERS':
            self.mutate_integer(neg=True)
        if partition == 'ZERO_INTEGER':
            self.mutate_integer(zero=True)
        if partition == 'LARGE_INTEGER':
            self.mutate_integer(large=True)
        if partition == 'NEGATIVE_LARGE_INTEGER':
            self.mutate_integer(neg_large=True)
        if partition == 'EMPTY':
            self.mutate_integer(empty=True)
        if partition == 'NONE':
            self.mutate_integer(none=True)
        if partition == 'NAN':
            self.mutate_integer(nan=True)

        # FLOAT
        if partition == 'NEGATIVE_FLOAT':
            self.mutate_float(neg=True)
        if partition == 'ZERO_FLOAT':
            self.mutate_float(zero=True)
        if partition == 'LARGE_FLOAT':
            self.mutate_float(large=True)
        if partition == 'NEGATIVE_LARGE_FLOAT':
            self.mutate_float(neg_large=True)
        if partition == 'EMPTY':
            self.mutate_float(empty=True)
        if partition == 'NONE':
            self.mutate_float(none=True)
        if partition == 'NAN':
            self.mutate_float(nan=True)

        # STRING

        if partition == 'INVALID_STRING':
            self.mutate_str(invalid=True)
        if partition == 'EMPTY_STRING':
            self.mutate_str(empty1=True)
        if partition == 'EMPTY':
            self.mutate_str(empty2=True)
        if partition == 'NONE':
            self.mutate_str(none=True)
        if partition == 'NAN':
            self.mutate_str(nan=True)
            
        
        if partition == 'RANDOM_BOOL':
            self.mutate_bool(random_=True)
        if partition == 'NONE_BOOL':
            self.mutate_bool(none=True)
        if partition == 'NAN_BOOL':
            self.mutate_bool(nan=True)
        if partition == 'EMPTY_BOOL':
            self.mutate_bool(empty=True)
        if partition == 'ZERO_BOOL':
            self.mutate_bool(zero=True)

class TorchAPI(API):
    def __init__(self, api_name, record=None):
        super().__init__(api_name)
        if record == None:
            record = TorchDatabase.get_rand_record(self.api)
        self.args = self.generate_args_from_record(record)
        self.is_class = inspect.isclass(eval(self.api))

    def each_arg_mutate(self, arg, partition):
        if do_type_mutation():
            arg.mutate_type()
        arg.each_arg_mutation(partition)

    def new_mutate_multiple(self, arg, r):
        if do_type_mutation():
            arg.mutate_type()
        arg.new_mutation_multiple(r)

    def new_mutate_torch(self):
        for p in self.args:
            arg = self.args[p]
            if do_type_mutation():
                arg.mutate_type()
            arg.new_mutation()

    def mutate(self, enable_value=True, enable_type=True, enable_db=True):
        num_arg = len(self.args)
        if num_arg == 0:
            return
        num_Mutation = randint(1, num_arg + 1)
        for _ in range(num_Mutation):
            arg_name = choice(list(self.args.keys()))
            arg = self.args[arg_name]

            if enable_type and do_type_mutation():
                arg.mutate_type()
            do_value_mutation = True
            if enable_db and do_select_from_db():
                new_arg, success = TorchDatabase.select_rand_over_db(
                    self.api, arg_name)
                if success:
                    new_arg = TorchArgument.generate_arg_from_signature(
                        new_arg)
                    self.args[arg_name] = new_arg
                    do_value_mutation = False
            if enable_value and do_value_mutation:
                arg.mutate_value()

    def my_to_code(self,
                   prefix="arg",
                   res="res",
                   is_cuda=False,
                   use_try=False,
                   error_res=None,
                   low_precision=False) -> str:
        code = ""
        arg_str = ""
        count = 1

        for key, arg in self.args.items():
            if key == "input_signature":
                continue
            arg_name = f"{prefix}_{count}"
            code += arg.to_code(arg_name,
                                low_precision=low_precision,
                                is_cuda=is_cuda)
            if key.startswith("parameter:"):
                arg_str += f"{arg_name},"
            else:
                arg_str += f"{key}={arg_name},"
            count += 1

        res_code = ""
        if self.is_class:
            if is_cuda:
                code += f"{prefix}_class = {self.api}({arg_str}).cuda()\n"
            else:
                code += f"{prefix}_class = {self.api}({arg_str})\n"

            if "input_signature" in self.args.keys():
                arg_name = f"{prefix}_{count}"
                code += self.args["input_signature"].to_code(
                    arg_name, low_precision=low_precision, is_cuda=is_cuda)
                res_code = f"{res} = {prefix}_class(*{arg_name})\n"
        else:
            res_code = f"{res} = {self.api}({arg_str})\n"

        return res_code

    def to_code(
        self,
        prefix="arg",
        res="res",
        is_cuda=False,
        use_try=True,
        error_res=None,
        low_precision=False,
    ) -> str:
        code = ""
        arg_str = ""
        count = 1

        for key, arg in self.args.items():
            if key == "input_signature":
                continue
            arg_name = f"{prefix}_{count}"
            code += arg.to_code(arg_name,
                                low_precision=low_precision, is_cuda=is_cuda)
            if key.startswith("parameter:"):
                arg_str += f"{arg_name},"
            else:
                arg_str += f"{key}={arg_name},"
            count += 1

        res_code = ""
        if self.is_class:
            if is_cuda:
                code += f"{prefix}_class = {self.api}({arg_str}).cuda()\n"
            else:
                code += f"{prefix}_class = {self.api}({arg_str})\n"

            if "input_signature" in self.args.keys():
                arg_name = f"{prefix}_{count}"
                code += self.args["input_signature"].to_code(
                    arg_name, low_precision=low_precision, is_cuda=is_cuda
                )
                res_code = f"{res} = {prefix}_class(*{arg_name})\n"
        else:
            res_code = f"{res} = {self.api}({arg_str})\n"

        return code + self.invocation_code(
            res, error_res, res_code, use_try, low_precision
        )

    def to_diff_code(
        self,
        oracle: OracleType,
        prefix="arg",
        res="res",
        *,
        error_res=None,
        use_try=True,
    ) -> str:
        """Generate code for the oracle"""
        code = ""
        arg_str = ""
        count = 1

        for key, arg in self.args.items():
            if key == "input_signature":
                continue
            arg_name = f"{prefix}_{count}"
            code += arg.to_diff_code(arg_name, oracle)
            if key.startswith("parameter:"):
                arg_str += f"{arg_name},"
            else:
                arg_str += f"{key}={arg_name},"
            count += 1

        res_code = ""
        if self.is_class:
            if oracle == OracleType.CUDA:
                code = f"{prefix}_class = {prefix}_class.cuda()\n"
            if "input_signature" in self.args.keys():
                arg_name = f"{prefix}_{count}"
                code += self.args["input_signature"].to_diff_code(
                    arg_name, oracle)
                res_code = f"{res} = {prefix}_class(*{arg_name})\n"
        else:
            res_code = f"{res} = {self.api}({arg_str})\n"

        return code + self.invocation_code(
            res, error_res, res_code, use_try, oracle == OracleType.PRECISION
        )

    @staticmethod
    def invocation_code(res, error_res, res_code, use_try, low_precision):
        code = ""
        if use_try:
            # specified with run_and_check function in relation_tools.py
            if error_res == None:
                error_res = res
            temp_code = "try:\n"
            temp_code += API.indent_code(res_code)
            temp_code += f'except Exception as e:\n  print("Error:"+str(e))\n'
            res_code = temp_code

        if low_precision:
            code += "start = time.time()\n"
            code += res_code
            code += f"{res} = time.time() - start\n"
        else:
            code += res_code
        return code

    @staticmethod
    def generate_args_from_record(record: dict) -> dict:
        args = {}
        for key in record.keys():
            if key != "output_signature":
                args[key] = TorchArgument.generate_arg_from_signature(
                    record[key])
        return args