csvs = [
    [R"../models/SRDeblur_cifar_no_lstm/SRDeblur_cifar_no_lstm.log.csv",
     ['loss', 'scale32_output_ssim_metric'],
     "SRNDeblur_cifar_no_lstm"],
    [R"../models/SRDeblur_cifar_lstm/SRDeblur_cifar_lstm.log.csv",
     ['loss', 'scale32_output_ssim_metric'],
     "SRNDeblur_cifar_lstm"],
    [R"../models/SRDeblur_reds_hr/SRDeblur_reds_hr.log.csv",
     ['loss', 'scale256_output_ssim_metric'],
     "SRNDeblur_reds"],
    [R"../models/EDDenseNet/EDDenseNet.log.csv", ["loss", "ssim_metric"], "EDDenseNet"],
    [R"../models/ResUNet1/ResUNet1.log.csv", ["loss", "ssim_metric"], "ResUnet1"],
    [R"../models/ResUNet3/ResUNet3.log.csv", ["loss", "ssim_metric"], "ResUnet3"],
    [R"../models/CAESSC_d30_f64/CAESSC_d30_f64.log.csv", ["loss", "ssim_metric"], "CAESSC_d30_f64"],
    [R"../models/CAESSC_d22_f128_half/CAESSC_d22_f128_half.log.csv", ["loss", "ssim_metric"], "CAESSC_d22_f128_half"],
    [R"../models/CAESSC_d22_f128_half_no_sigmoid/CAESSC_d22_f128_half_no_sigmoid.log.csv", ["loss", "ssim_metric"], "CAESSC_d22_f128_half_no_sigmoid"]
]

from basemodel.common import plot_csv

for csv in csvs:
    plot_csv(csv[0], csv[1], csv[2])
