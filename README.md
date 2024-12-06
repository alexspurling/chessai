
Instructions for installing pytorch are here: https://pytorch.org/get-started/locally/

Find out what version of CUDA your driver supports by running:

```commandline
nvidia-smi.exe
```

I installed it with support for CUDA 12.4 like this:

```commandline
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```