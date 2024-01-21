from  davarocr.davar_common.apis import inference_model, init_model
import torch
from mmdet.datasets.pipelines import Compose
import numpy as np
import cv2
from mmcv.parallel import collate, scatter
import os
root_path = '/'.join(__file__.split('/')[:-1])

config_path = os.path.join(root_path,'config_lpgma.py')
model_path = os.path.join(root_path,'maskrcnn-lgpma-pub-e12-pub.pth')
img_path = os.path.join(root_path,'test.jpg')

def test_model():
    global config_path, model_path, img_path 

    model = init_model(config_path,model_path)
    imgs = {'img':cv2.imread(img_path)}
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    data = imgs
    cfg.data.test.pipeline[1]['transforms'][1]['mean'] = \
            np.array(cfg.data.test.pipeline[1]['transforms'][1]['mean'])
    cfg.data.test.pipeline[1]['transforms'][1]['std'] = \
            np.array(cfg.data.test.pipeline[1]['transforms'][1]['std'])
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    device = int(str(device).split(":")[-1])
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    assert len(result[0].keys()) == 2, "wrong result"
    assert len(result[0]['content_ann'].keys()) == 2, "results don't have content_ann"

def test_inference_model():
    global config_path, model_path, img_path 
    model = init_model(config_path,model_path)
    imgs = {'img':cv2.imread(img_path)}
    result = inference_model(model,imgs)
    assert len(result[0].keys()) == 2, "wrong result"
    assert len(result[0]['content_ann'].keys()) == 2, "results don't have content_ann"

