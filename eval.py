from detectron2.modeling import build_model
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
import sys
import os

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

# Load the config
cfg = get_cfg()
cfg.merge_from_file(config_file)
model = build_model(cfg)
cfg.OUTPUT_DIR = out_dir

print('----------BEST MODEL METRICS:--------------:')
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS) # Load the best model
predictor = DefaultPredictor(cfg)
out_best = os.path.join(cfg.OUTPUT_DIR, "output_best")
evaluator = COCOEvaluator("SewerDataValidation", ("bbox",), False, output_dir=out_best)
val_loader = build_detection_test_loader(cfg, "SewerDataValidation")
print(inference_on_dataset(model, val_loader, evaluator))

# Evaluate testing data on the final model (all iterations) - will likely be worse
print('----------FINAL MODEL METRICS:-------------:')
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set a custom testing threshold
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS) # Load the final/last model
predictor = DefaultPredictor(cfg)
out_final = os.path.join(cfg.OUTPUT_DIR, "output_final")
evaluator = COCOEvaluator("SewerDataValidation", ("bbox",), False, output_dir=out_final)
val_loader = build_detection_test_loader(cfg, "SewerDataValidation")
print(evaluator)
print(val_loader)
print(inference_on_dataset(model, val_loader, evaluator))


