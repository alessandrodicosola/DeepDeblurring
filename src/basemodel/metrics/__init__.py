from src.basemodel.metrics.psnr import  psnr_metric
from src.basemodel.metrics.ssim import ssim_metric
#from src.basemodel.metrics.q import q_metric

# Metrics to import in all models
metrics = [ "mse", psnr_metric, ssim_metric ]
metrics_dict = { (metric.__name__ if not isinstance(metric,str) else metric):metric for metric in metrics}
