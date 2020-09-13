from src.models.reds.SRNDeblur import SRNDeblur as SRNDeblur_reds
from src.models.cifar10.SRDeblur import SRDeblur as SRNDeblur_cifar
from src.models.cifar10.ResUNet import ResUNet
from src.models.cifar10.CAESSC import CAESSC
from src.models.cifar10.EDDenseNet import EDDenseNet

"""
Possibile models:
    models = [CAESSC(), EDDenseNet(), ResUNet(1), ResUNet(3), SRNDeblur_cifar(use_lstm=False),
        SRNDeblur_cifar(use_lstm=True),
         
        low_res affect only test(); training is done on high res images
        SRNDeblur_reds(low_res=True)  
        SRNDeblur_reds(low_res=False)
"""
"""
Example:

model = SRNDeblur_reds(low_res=True)
model.train()
model.test()
model.evaluate()
"""

for model in [CAESSC(16, 128, True), CAESSC(16, 128, False), CAESSC(32, 64, True), CAESSC(32, 64, False)]:
    model.train()
    model.evaluate()
    model.test()
