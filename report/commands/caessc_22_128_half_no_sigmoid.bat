@echo off
cd /D "D:\University\PROJECTS\DL\deeplearning-deblurring"
call activate deep-learning
python src/main_cli.py CAESSC 22 128 --downsample --use_relu