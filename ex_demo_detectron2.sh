#!/bin/bash
#SBATCH -J eval-detect2
#SBATCH -p normal_q
#SBATCH -N 1
#SBATCH -t 59:00
#SBATCH -o out_eval_detect2
python ./detectron2/demo/demo.py --config-file ./faster_rcnn_R_50_FPN_3x.yaml \
    --input ./input_images/*.jpg \
    --output ./output_images \
    --confidence-threshold 0.000002 \
    --opts MODEL.DEVICE cpu MODEL.WEIGHTS ./model_final.pth
