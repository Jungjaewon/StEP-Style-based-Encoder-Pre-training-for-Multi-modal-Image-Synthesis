
import torchvision.models as models
import torch.nn as nn
import torch

if __name__ == '__main__':

    pass
    """
    res = models.resnet18(pretrained=True)
    res = list(res.children())[:-1]
    _res = nn.Sequential(*res)
    print(_res)

    tensor = torch.randn((1,3,256,256))
    print(_res(tensor).size())
    """

    """
    a = torch.randn((1,10,1,1))
    _, a_ch, _, _ = a.size()
    b = torch.randn((1,3,10,10))
    batch, ch, h, w = b.size()
    a = a.expand(batch, a_ch, h, w)
    print(a.size())
    result = torch.cat([a,b], dim=1)
    print(result.size())
    """

    tensor = torch.randn(4,5)

    tensor = tensor.unsqueeze(-1).unsqueeze(-1)
    print(tensor.size())
