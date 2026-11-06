"""Persistent VIPLite network through our C ABI, compiled on the target board."""
import ctypes as C
from pathlib import Path
import threading
import numpy as np


class TensorInfo(C.Structure):
    _fields_ = [('dimensions', C.c_uint32), ('sizes', C.c_uint32*6),
                ('format', C.c_uint32), ('quantization', C.c_uint32),
                ('fixed_point', C.c_int32), ('zero_point', C.c_int32),
                ('scale', C.c_float), ('bytes', C.c_uint32), ('elements', C.c_uint32)]

    @property
    def shape(self):
        return tuple(self.sizes[:self.dimensions])[::-1]

    def describe(self):
        return dict(shape=self.shape, format=self.format, quantization=self.quantization,
                    scale=self.scale, zero_point=self.zero_point,
                    fixed_point=self.fixed_point, bytes=self.bytes, elements=self.elements)


class VipNetwork:
    """Create, execute and destroy on the same inference thread."""
    def __init__(self, model, library=None):
        self.owner = threading.get_ident()
        self.handle = None
        path = Path(library) if library else Path(__file__).resolve().parent/'native/libfacelock_vip.so'
        if not Path(model).is_file():
            raise RuntimeError(f'Missing NPU NBG model: {model}')
        if not path.is_file():
            raise RuntimeError(f'Missing NPU bridge: {path}; run bash native/build.sh on the board')
        self.lib = C.CDLL(str(path))
        self.lib.fc_open.argtypes = [C.c_char_p, C.c_char_p, C.c_uint32]
        self.lib.fc_open.restype = C.c_void_p
        self.lib.fc_close.argtypes = [C.c_void_p]
        self.lib.fc_close.restype = None
        self.lib.fc_output_count.argtypes = [C.c_void_p]
        self.lib.fc_output_count.restype = C.c_uint32
        self.lib.fc_info.argtypes = [C.c_void_p, C.c_int, C.c_uint32, C.POINTER(TensorInfo)]
        self.lib.fc_info.restype = C.c_int
        self.lib.fc_run.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32]
        self.lib.fc_run.restype = C.c_int
        self.lib.fc_read_output.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint32]
        self.lib.fc_read_output.restype = C.c_int
        self.lib.fc_person_candidates.argtypes = [C.c_void_p,C.c_float,C.c_void_p,C.c_uint32]
        self.lib.fc_person_candidates.restype = C.c_int
        error = C.create_string_buffer(512)
        self.handle = self.lib.fc_open(str(model).encode(), error, len(error))
        if not self.handle:
            raise RuntimeError('NPU initialization: '+error.value.decode(errors='replace'))
        try:
            self.input_info = self._info(False, 0)
            self.output_info = [self._info(True, i) for i in range(self.lib.fc_output_count(self.handle))]
            self.outputs = [np.empty(info.shape, np.float32) for info in self.output_info]
            self.candidates = np.empty((25200,9),np.float32)
        except BaseException:
            self.close()
            raise

    def _info(self, output, index):
        info = TensorInfo()
        self._check(self.lib.fc_info(self.handle, int(output), index, C.byref(info)), 'query tensor')
        return info

    @staticmethod
    def _check(code, operation):
        if code:
            raise RuntimeError(f'VIPLite {operation} failed: {code}')

    def run(self, tensor):
        self._execute(tensor)
        for index, output in enumerate(self.outputs):
            self._check(self.lib.fc_read_output(self.handle, index, output.ctypes.data, output.size), 'read output')
        return self.outputs

    def _execute(self, tensor):
        if threading.get_ident() != self.owner or not self.handle:
            raise RuntimeError('Use VIPLite on its owning inference thread while open')
        if tensor.dtype != np.uint8 or not tensor.flags.c_contiguous or tensor.nbytes != self.input_info.bytes:
            raise ValueError('NPU input must match the queried uint8 size and contiguous CHW layout')
        self._check(self.lib.fc_run(self.handle, tensor.ctypes.data, tensor.nbytes), 'run')

    def run_people(self, tensor, threshold):
        self._execute(tensor)
        count = self.lib.fc_person_candidates(self.handle, threshold,
                                              self.candidates.ctypes.data, len(self.candidates))
        if count < 0:
            self._check(count,'person compaction')
        return self.candidates[:count]

    def close(self):
        if self.handle:
            if threading.get_ident() != self.owner:
                raise RuntimeError('Close VIPLite on its owning inference thread')
            self.lib.fc_close(self.handle)
            self.handle = None
