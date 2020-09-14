from src.models.reds.SRNDeblur import SRNDeblur as SRNDeblur_reds
from src.models.cifar10.SRDeblur import SRDeblur as SRNDeblur_cifar
from src.models.cifar10.ResUNet import ResUNet
from src.models.cifar10.CAESSC import CAESSC
from src.models.cifar10.EDDenseNet import EDDenseNet

"""
Possibile models:
    CAESSC(args)
    EDDenseNet()
    ResUNet(args)
    SRNDeblur_cifar(args)

Example:

model = SRNDeblur_reds(low_res=True)
model.train()
model.test()
model.evaluate()
"""

model = CAESSC(22, 128, True, batch_size=200)
model.train()
model.evaluate()
model.test()
