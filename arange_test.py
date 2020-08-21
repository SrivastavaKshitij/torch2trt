import torch
class Foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.arange(0, x.size()[0])
if __name__ == "__main__":
    from torch2trt import torch2trt
    model = Foo().eval().cuda()
    x = torch.randn(10, device='cuda')
    x_trt = torch2trt(model, [x])
