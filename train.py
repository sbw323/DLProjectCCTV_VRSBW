import detectron2
import argparse
from detectron2.utils.logger import setup_logger
setup_logger()
import sys
import os
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetMapper, build_detection_test_loader
from LossEvalHook import *
from detectron2.checkpoint import DetectionCheckpointer

# Custom Trainer that will evaluate the loss on the validation data
# during training. (used to avoid overfitting the model)
# The model with the lowest validation loss will be saved to <output>/model_best.pth
# This and the LossEvalHook class were borrowed from:
# https://ortegatron.medium.com/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"detectron2_coco_eval")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks


# Grab run options
if (len(sys.argv) != 4):
    print("Usage: ./train <DATA_DIR> <CONFIG> <OUTPUT_DIR>")
    quit()
data_dir = sys.argv[1]
config_file = sys.argv[2]
out_dir = sys.argv[3]

# Load the datasets
register_coco_instances("SewerDataTrain", {},\
    os.path.join(data_dir, "train/_annotations.coco.json"), os.path.join(data_dir,"train"))
register_coco_instances("SewerDataValidation", {},\
    os.path.join(data_dir, "valid/_annotations.coco.json"), os.path.join(data_dir,"valid"))
register_coco_instances("SewerDataTest", {},\
    os.path.join(data_dir, "test/_annotations.coco.json"), os.path.join(data_dir,"test"))

# Load Detectron configs from file
cfg = get_cfg()
cfg.merge_from_file(config_file)

# Set some manual configs
cfg.DATASETS.TRAIN = ("SewerDataTrain",)
cfg.DATASETS.TEST = ("SewerDataValidation",) # Used for validation throughout training.
cfg.OUTPUT_DIR = out_dir
#cfg.SOLVER.STEPS = [] # do not decay learning rate

# If you change this, should also change SOLVER.BASE_LR and IMS_PER_BATCH
# See: https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html#training-evaluation-in-command-line
cfg.SOLVER.NUM_GPUS = 1
cfg.SOLVER.BASE_LR = 0.02 / 16
cfg.IMS_PER_BATCH = 2

# How often to test evaluation data (in iterations)
#cfg.TEST.EVAL_PERIOD = 100
cfg.SOLVER.MAX_ITER = (90000)

# Other setup stuff
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Run the training
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# -------------------------------------------------------------------
# ---------------------------- TESTING ------------------------------
# -------------------------------------------------------------------
# Evaluate testing data on the best model (lowest validation loss)
print('----------BEST MODEL METRICS:--------------:')
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth") 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
DetectionCheckpointer(trainer.model).load(cfg.MODEL_WEIGHTS) # Load the best model
predictor = DefaultPredictor(cfg)
out_best = os.path.join(cfg.OUTPUT_DIR, "output_best")
evaluator = COCOEvaluator("SewerDataTest", ("bbox",), False, output_dir=out_best)
val_loader = build_detection_test_loader(cfg, "SewerDataTest")
print(inference_on_dataset(trainer.model, val_loader, evaluator))

# Evaluate testing data on the final model (all iterations) - will likely be worse
print('----------FINAL MODEL METRICS:-------------:')
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set a custom testing threshold
DetectionCheckpointer(trainer.model).load(cfg.MODEL_WEIGHTS) # Load the final/last model
predictor = DefaultPredictor(cfg)
out_final = os.path.join(cfg.OUTPUT_DIR, "output_final")
evaluator = COCOEvaluator("SewerDataTest", ("bbox",), False, output_dir=out_final)
val_loader = build_detection_test_loader(cfg, "SewerDataTest")
print(inference_on_dataset(trainer.model, val_loader, evaluator))


