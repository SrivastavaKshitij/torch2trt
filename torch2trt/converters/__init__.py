# dummy converters throw warnings method encountered
import tensorrt as trt
from .dummy_converters import *

# supported converters will override dummy converters

from .AdaptiveAvgPool2d import *
from .BatchNorm1d import *
from .BatchNorm2d import *
from .Conv import *
from .Conv1d import *
from .Conv2d import *
from .ConvTranspose import *
from .ConvTranspose2d import *
from .Identity import *
from .Linear import *
from .LogSoftmax import *
from .ReLU import *
from .ReLU6 import *
from .activation import *
from .adaptive_avg_pool2d import *
from .adaptive_max_pool2d import *
from .add import *
from .avg_pool import *
from .batch_norm import *
from .cat import *
from .chunk import *
from .clamp import *
from .compare import *
from .div import *
from .getitem import *
from .identity import *
from .instance_norm import *
from .interpolate import *
from .max import *
from .max_pool2d import *
from .mean import *
from .min import *
from .mul import *
from .normalize import *
from .narrow import *
from .pad import *
from .permute import *
from .pow import *
from .prelu import *
from .prod import *
from .relu import *
from .relu6 import *
from .sigmoid import *
from .softmax import *
from .split import *
from .stack import *
from .sub import *
from .sum import *
from .tanh import *
from .transpose import *
from .unary import *
from .view import *

from .QATConvBNRelu2D import *

