
import cv2, torch
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import meta_arch
from detectron2.utils.visualizer import Visualizer
from torch import Tensor
from detectron2.structures import Boxes
from typing import Dict, List, Tuple
from detectron2.modeling import build_model, GeneralizedRCNN
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.file_io import PathManager
from detectron2.export.torchscript import dump_torchscript_IR
from detectron2.export.torchscript import scripting_with_instances
from detectron2.utils.visualizer import ColorMode
from detectron2.export.caffe2_modeling import assemble_rcnn_outputs_by_name
from time import perf_counter, process_time
import time

setup_logger()

#   Config setup...
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = get_cfg()
model_config = "./config.yaml"
cfg.merge_from_file(model_config)
cfg.MODEL.DEVICE = device


def ts_warmup(model, inputs):
  #   Waiting for system stability
  for i in range(10):
      model(inputs)[0]
  #   Calculate inference time 
  start = time.time()
  results = model(inputs)
  elaps_time = time.time() - start
  return elaps_time, results


def tracing_infer(img_name):
    org_img = cv2.resize(cv2.imread(img_name), (400, 300))
    #   Load TorchScript model
    ts_model = torch.jit.load('model.ts')
    ts_model = ts_model.to(device).eval()

    height, width = org_img.shape[:2]
    image_sizes = [(height, width)]

    with torch.no_grad():
        img = torch.as_tensor(org_img.astype("float32").transpose(2, 0, 1)).to(device)
        elaps_time, results = ts_warmup(ts_model, img)
        """ 
            =========== Parse torchscript outputs ============
            results: [{
                        'pred_boxes': float_Tensor,
                        'scores': float_Tensor,
                        'pred_classes': int_Tensor,
                        'pred_masks': float_Tensor
                    }]
        """
        
        #   Visualize bounding box
        print("==================================")
        print(f" >> Elaps time: {elaps_time:.3f}s.")
        print(f" >> FPS: {1/elaps_time:.1f}")

        for coor in results[0]:
            org_img = cv2.rectangle(org_img, (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3])), (255, 0, 0), 2)
        print(org_img.shape)
        plt.imshow(org_img)
        plt.axis("off")
        plt.savefig("image.png")
        plt.show()

        #   Visualiza bounding box and segmentation mask
        # output_custom = {
        #                     "bbox_nms": results[0],
        #                     "score_nms": results[3],
        #                     "class_nms": results[1],
        #                     "mask_fcn_probs": results[2]
        #                 }

        # results = assemble_rcnn_outputs_by_name(image_sizes, output_custom, force_mask_on=True)
        # inputs = [{"image": img, "height": height, "width": width}]

        # # Replace with the used model
        # outputs = getattr( meta_arch, cfg.MODEL.META_ARCHITECTURE)._postprocess(results, inputs, image_sizes )

        # print("==================================")
        # print(" >> Size:", img.shape)
        # print(f" >> Elaps time: {elaps_time:.3f}s.")
        # print(f" >> FPS: {1/elaps_time:.1f}")
            
        # #   Visualization
        # v = Visualizer(org_img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        # out = v.draw_instance_predictions(outputs[0]["instances"].to("cpu"))
        # img = out.get_image()[:, :, ::-1]

        # #   Save image
        # cv2.imwrite("results/img.jpg", img)


if __name__ == "__main__":
    img_sizes = [(200, 200), (300, 300), (400, 400), (500, 500), (600, 600), (800, 800),
             (300, 200), (400, 200), (400, 300), (600, 300), (800, 300), (800, 400)]    

    tracing_infer("grape.jpg")

""" 
import torch
import torch.nn as nn
import torch.nn.functional as F

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# model = Net()
# model = torch.jit.load("./model.ts")
model = torch.jit.load('./model.ts')
model = model.to("cuda").eval()
print(model) """