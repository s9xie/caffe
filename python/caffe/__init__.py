from .pycaffe import Net, SGDSolver
from ._caffe import set_mode_cpu, set_mode_gpu, set_phase_train, \
        set_phase_test, set_device
from .classifier import Classifier
from .detector import Detector
import io
