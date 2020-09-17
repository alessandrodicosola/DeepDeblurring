# DeepDeblurring
DeepDebluring project for the course [91250] Deep Learning at UniBo

# Requirements 

- python 3.x

## Conda
- ```python
  conda create --name <env> --file requirements_conda.txt
  ```
- use directly requirements_conda.yml within anaconda
## Pip
```python
pip install -r requirements_pip.txt
```
# Usage

- Go to the project folder (where src, report andmodels folders are)
- activate environment
- ```python
  python main_cli.py MODEL PARAMS
  ```

# Documentation

```python
python main_cli.py -h
python main_cli.py MODEL -h 
```

# Report
The report is report/***Report.pdf***

The powerpoint presentation is report/***powerpoint.pdf***

# Models
The models implemented are:
- [ResUNet](/src/models/cifar10/ResUNet.py)
- [EDDenseNet](/src/models/cifar10/EDDenseNet.py)
- [CAESSC](/src/models/cifar10/CAESSC.py)
- [SRNDeblur_cifar](/src/models/cifar10/SRNDeblur.py)
- [SRNDeblur_reds](/src/models/reds/SRNDeblur.py)


The model and weights are save in **/models/[MODEL NAME]/[MODEL NAME]_best.h5**