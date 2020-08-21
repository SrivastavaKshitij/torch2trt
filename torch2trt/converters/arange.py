from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test
import math
import tensorrt as trt 

@tensorrt_converter('torch.arange')
def convert_cat(ctx):
    start = get_arg(ctx, 'start', pos=0, default=0)
    end  = get_arg(ctx, 'end', pos=1)
    step  = get_arg(ctx, 'step', pos=2, default=1)
    dtype  = get_arg(ctx, 'dtype',  pos=4)

    output = ctx.method_return
    num_steps = math.floor((end - start) / step)

    layer = ctx.network.add_fill(shape=trt.Dims([num_steps]), op=trt.tensorrt.LINSPACE)
    layer = ctx.network.add_concatenation(inputs=trt_inputs)
    layer.axis = dim - 1
    output._trt = layer.get_output(0)

class arange(torch.nn.Module):
    def __init__(self, length):
        super(arange, self).__init__()

    def forward(self, x):
        return torch.arange(0, x.size()[0])

@add_module_test(torch.float32, torch.device('cuda'), [(4)])
def test_arange_basic():
    return arange()
