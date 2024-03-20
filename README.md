# MTSC-Net

Code of the paper MTSC-Net: A Semi-Supervised Counting Network for Estimating the Number of Slash Pine New Shoots.
***
# Running Environment
```
PyCharm 2022
PyTorch 1.11.0
Intel(R) Core(TM) i9-12900K
NVIDIA GeForce RTX 3090
the operating system was Windows 10
and the CUDA version is 11.6
```
# Run:
python main.py

# Data Processing:
follow the file "process_new_shoots.py" to produce the ground-truth density map, file.mat to file.h5

# Requirments
```
h5py==3.10.0	
numpy==1.22.4	
onnx==1.15.0	
opencv-python==4.9.0.80	
pillow==9.5.0	
python==3.8.18	
pytorch==1.13.0	
scikit-image==0.21.0	
scipy==1.10.1	
timm==0.9.12	
torchaudio==0.13.0	
torchvision==0.14.0	
tqdm==4.66.1	
```

