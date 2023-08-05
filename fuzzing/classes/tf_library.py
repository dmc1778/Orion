import contextlib
from io import StringIO
import sys
import subprocess
from constants.keys import (
    ERR_CPU_KEY,
    ERR_GPU_KEY,
    ERR_HIGH_KEY,
    ERR_LOW_KEY,
    ERROR_KEY,
    RES_CPU_KEY,
    RES_GPU_KEY,
    TIME_HIGH_KEY,
    TIME_LOW_KEY,
)
from constants.enum import OracleType
from classes.database import TFDatabase
from classes.library import Library
from classes.tf_api import TFAPI, TFArgument
from classes.argument import Argument, ArgType
import re
import numpy as np
import time
import tensorflow as tf
from os.path import join
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def runProcess(exe):
    p = subprocess.Popen(
        exe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
    )
    while True:
        # returns None while subprocess is running
        retcode = p.poll()
        line = p.stdout.readline()
        yield line
        if retcode is not None:
            break


class TFLibrary(Library):
    def __init__(
        self, output_dir, diff_bound=1e-5, time_bound=10, time_thresold=1e-3
    ) -> None:
        super().__init__(output_dir)
        self.diff_bound = diff_bound
        self.time_bound = time_bound
        self.time_thresold = time_thresold

    def test_with_oracle(self, api: TFAPI, oracle: OracleType):
        if oracle == OracleType.CRASH:
            # We need call another process to catch the crash error

            part_from = ".".join(api.api.split(".")[0:-2])

            code = "import tensorflow as tf\n"
            code += "import os\n"
            code += "import numpy as np\n"
            # code += 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"\n'

            if re.findall(r"(tensorflow\.python)", api.api):
                code += f"from {part_from} import {api.api.split('.')[-2]}\n"
                api.api = ".".join(api.api.split(".")[-2:])
                code += self.generate_code(api, oracle)
            else:
                code += self.generate_code(api, oracle)

            write_code = "results = dict()\n" + code + "\nprint(results)\n"

            with open(join(self.directory, f"temp.py"), "w") as f:
                f.write(write_code)

            results, error = self.run_code(write_code)
            if self.is_crash_msg(error):
                self.write_to_dir(
                    join(self.output[oracle], "potential-bug"), api.api, code
                )
            # if error == None:
            #     self.write_to_dir(
            #         join(self.output[oracle], "success"), api.api, write_code)
            # else:
            #     self.write_to_dir(
            #         join(self.output[oracle], "fail"), api.api, write_code)
        elif oracle == OracleType.CUDA:
            part_from = ".".join(api.api.split(".")[0:-2])

            code = "import tensorflow as tf\n"

            code += "import numpy as np\n"
            if re.findall(r"(tensorflow\.python)", api.api):
                code += f"from {part_from} import {api.api.split('.')[-2]}\n"
                api.api = ".".join(api.api.split(".")[-2:])
                code += self.generate_code(api, oracle)
            else:
                code += self.generate_code(api, oracle)

            write_code = "results = dict()\n" + code + "\nprint(results)\n"
            with open(join(self.directory, f"temp.py"), "w") as f:
                f.write(write_code)

            results, error = self.run_code(write_code)
            # err_cpu = results[ERR_CPU_KEY]
            # err_gpu = results[ERR_GPU_KEY]
            # write_dir = ""
            # if error is None:
            #     if (err_cpu is None) != (err_gpu is None):
            #         write_dir = join(self.output[oracle], "potential-bug")
            #     elif err_cpu == None:
            #         res_cpu = results[RES_CPU_KEY]
            #         res_gpu = results[RES_GPU_KEY]
            #         if self.is_equal(res_cpu, res_gpu):
            #             # write_dir = join(self.output[oracle], "success")
            #             pass
            #         else:
            #             write_dir = join(self.output[oracle], "potential-bug")
            #     elif "SystemError" in err_cpu or "SystemError" in err_gpu:
            #         write_dir = join(self.output[oracle], "potential-bug")
            #     else:
            #         # write_dir = join(self.output[oracle], "success")
            #         pass
            # elif "SystemError" in error:
            #     write_dir = join(self.output[oracle], "potential-bug")
            # else:
            #     # write_dir = join(self.output[oracle], "fail")
            #     pass
            # self.write_to_dir(write_dir, api.api, write_code)
        elif oracle == OracleType.PRECISION:

            part_from = ".".join(api.api.split(".")[0:-2])

            code = "import tensorflow as tf\n"
            code += "import time\n"
            if re.findall(r"(tensorflow\.python)", api.api):
                code += f"from {part_from} import {api.api.split('.')[-2]}\n"
                api.api = ".".join(api.api.split(".")[-2:])
                code += self.generate_code(api, oracle)
            else:
                code += self.generate_code(api, oracle)

            write_code = "results = dict()\n" + code
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(write_code)

            results, error = self.run_code(write_code)
            err_high = results[ERR_HIGH_KEY]
            err_low = results[ERR_LOW_KEY]
            write_dir = ""
            if error is None:
                if (err_high is None) != (err_low is None):
                    write_dir = join(self.output[oracle], "potential-bug")
                elif err_high == None:
                    time_high = results[TIME_HIGH_KEY]
                    time_low = results[TIME_LOW_KEY]
                    if (
                        time_low >= self.time_bound * time_high
                        and time_high >= self.time_thresold
                    ):
                        write_dir = join(self.output[oracle], "potential-bug")
                    else:
                        write_dir = join(self.output[oracle], "success")
                elif "SystemError" in err_high or "SystemError" in err_low:
                    write_dir = join(self.output[oracle], "potential-bug")
                else:
                    write_dir = join(self.output[oracle], "success")
            elif "SystemError" in error:
                write_dir = join(self.output[oracle], "potential-bug")
            else:
                write_dir = join(self.output[oracle], "fail")
            self.write_to_dir(write_dir, api.api, write_code)

    @staticmethod
    def generate_code(api: TFAPI, oracle: OracleType) -> str:
        code = ""
        if oracle == OracleType.CRASH:
            code += api.to_code_oracle(oracle=oracle)
            return code
        elif oracle == OracleType.CUDA:
            code += api.to_code_oracle(oracle=oracle)
            return code
        elif oracle == OracleType.PRECISION:
            code += api.to_code_oracle(oracle=oracle)
            return code
        else:
            assert 0

    @staticmethod
    def run_code(code):
        results = dict()
        results[ERR_CPU_KEY] = None
        results[ERR_GPU_KEY] = None
        results[ERR_HIGH_KEY] = None
        results[ERR_LOW_KEY] = None
        results[RES_CPU_KEY] = None
        results[RES_GPU_KEY] = None
        results[TIME_HIGH_KEY] = None
        results[TIME_LOW_KEY] = None

        exec(code)
        error = results[ERROR_KEY] if ERROR_KEY in results else None
        return results, error

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res != None:
            return res
        if isinstance(x, tf.Tensor):
            return ArgType.TF_TENSOR
        elif isinstance(x, tf.DType):
            return ArgType.TF_DTYPE
        else:
            return ArgType.TF_OBJECT

    @staticmethod
    def _eval_k(x):
        return tf.convert_to_tensor(x).numpy()

    @staticmethod
    def get_tensor_value(t):
        if isinstance(t, tf.SparseTensor):
            return tf.sparse.to_dense(t).numpy()
        else:
            return t.numpy()
        
    @staticmethod
    def is_crash_msg(error_msg):
        if error_msg == None:
            return False
        if "INTERNAL ASSERT" in error_msg:
            return True
        elif "Segmentation fault" in error_msg:
            return True
        elif "Aborted" in error_msg:
            return True
        elif "Killed" in error_msg:
            return True
        elif "Floating point exception" in error_msg:
            return True
        elif "core dumped" in error_msg:
            return True
        else:
            return False


    @staticmethod
    def is_equal(x, y):
        x_type = TFArgument.get_type(x)
        y_type = TFArgument.get_type(y)
        if x_type != y_type:
            return False
        if x_type == ArgType.KERAS_TENSOR:
            return tf.math.equal(x, y)
        if x_type == ArgType.TF_TENSOR:
            try:
                if isinstance(x, tf.RaggedTensor) != isinstance(y, tf.RaggedTensor):
                    return False
                if isinstance(x, tf.RaggedTensor):
                    s = tf.math.equal(x, y)
                    return s.flat_values.numpy().all()
                np_x = TFLibrary.get_tensor_value(x)
                np_y = TFLibrary.get_tensor_value(y)
                if x.dtype.is_floating:
                    return tf.experimental.numpy.allclose(
                        np_x, np_y, rtol=1e-3, atol=1e-4
                    )
                elif x.dtype.is_integer:
                    return np.equal(np_x, np_y).all()
            except:
                raise ValueError(
                    f"Comparison between {type(x)} is not supported now.")
            return True
        elif x_type == ArgType.FLOAT:
            return abs(x - y) < 1e-5
        elif x_type in [ArgType.LIST, ArgType.TUPLE]:
            if len(x) != len(y):
                return False
            for i in range(len(x)):
                if TFLibrary.is_equal(x[i], y[i]) == False:
                    return False
            return True

        else:
            try:
                flag = x == y
            except:
                return True

            if isinstance(flag, np.ndarray):
                flag = flag.all()
            try:
                if flag:
                    pass
            except:
                flag = True
            return
