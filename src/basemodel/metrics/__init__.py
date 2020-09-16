from .psnr import psnr_metric
from .ssim import ssim_metric

# from .q import q_metric

# Metrics to import in all models
metrics = ["mse", psnr_metric, ssim_metric]
metrics_dict = {(metric.__name__ if not isinstance(metric, str) else metric): metric for metric in metrics}
