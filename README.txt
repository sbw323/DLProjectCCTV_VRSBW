# Google Colab Instructions:
# You gotta make sure you have the right version of PyTorch Matching cuda 11.1
# Detectron2 has to be built to work with that
# DONT USE TORCH 1.8 it has a problem that doesn't let it run!
# USE: Torch 1.9.0, torchvision 0.10.0
# TORCH:
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# Detectron2
# python -m pip install detectron2 -f \
# https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# Run it:
# ./train_runner.sh
