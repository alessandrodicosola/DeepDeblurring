### MODELS ###
##############
import argparse

from models.cifar10.CAESSC import CAESSC
from models.cifar10.EDDenseNet import EDDenseNet
from models.cifar10.ResUNet import ResUNet
from models.cifar10.SRNDeblur import SRNDeblur as SRNDeblur_cifar10
from models.reds.SRNDeblur import SRNDeblur as SRNDeblur_reds

available_arguments = {
    "ResUNet": [{"dest": "resblocks", "help": "Amount of ResBlock to use inside the architecture",
                 "choices": [1, 3], "type": int}],
    "EDDenseNet": {},
    "CAESSC": [{"dest": "depth", "help": "Total amount of layers", "type": int, "choices": [22, 30]},
               {"dest": "filters", "help": "Amount of filters to use in each layer", "type": int,
                "choices": [64, 128]},
               ["--downsample",
                {"help": "Allow downsampling on first layer",
                 "action": "store_true"}],
               ["--use_relu",
                {"help": "Use ReLU as last activation function",
                 "action": "store_true"}]],
    "SRNDeblur_cifar": [{"dest": "use_lstm", "action": "store_true"}],
    "SRNDeblur_reds": [{"dest": "low_res_test", "action": "store_true"}]
}

name = "Deep Deblurring"

args = argparse.ArgumentParser(prog="Deep deblurring")

subargs = args.add_subparsers(title="Available models", help="MODEL -h for additional information", dest="model")
for model in available_arguments:
    parser = subargs.add_parser(model,
                                description="Available configuration: << CAESSC 22 128 --downsample>> << CAESSC 30 64 >>, << CAESSC 22 128 --downsample --use_relu >>" if model == "CAESSC" else None)
    for arg in available_arguments[model]:
        if isinstance(arg,list):
            # arg[0]: positional arguments
            # arg[1]: named arguments
            parser.add_argument(arg[0], **(arg[1]))
        elif isinstance(arg,dict):
            # only named arguments
            parser.add_argument(**arg)
        else:
            raise RuntimeError("Invalid data structure")

result = args.parse_args()

model = None
if result.model == "ResUNet":
    model = ResUNet(result.resblocks)
elif result.model == "EDdenseNet":
    model = EDDenseNet()
elif result.model == "CAESSC":
    model = CAESSC(result.depth, result.filters, result.downsample, not result.use_relu)
elif result.model == "SRNDeblur_cifar":
    model = SRNDeblur_cifar10(result.use_lstm)
elif result.model == "SRNDeblur_reds":
    model = SRNDeblur_reds(result.low_res_test)
else:
    raise RuntimeError(f"invalid model: {result.model}")

model.test()
model.open_test()
