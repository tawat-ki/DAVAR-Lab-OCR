from  davarocr.davar_common.apis import inference_model, init_model
import cv2
from mmcv.parallel import collate, scatter

config_path = "/app/DAVAR-Lab-OCR/demo/table_recognition/lgpma/configs/ocr_models/rcg_res32_bilstm_attn_pubtabnet_sensitive.py"
model_path = "/app/model.pth"
img = {'img':cv2.imread("/app/DAVAR-Lab-OCR/demo/table_recognition/lgpma/vis/PMC3160368_005_00.png")}
model = init_model(config_path,model_path)
def inference_model(model, imgs):
    """ Inference image(s) with the models
        Model types can be 'DETECTOR'(default), 'RECOGNIZOR', 'SPOTTER', 'INFO_EXTRACTOR'

    Args:
        model (nn.Module): The loaded model
        imgs (str | nd.array | list(str|nd.array)): Image files. It can be a filename of np array (single img inference)
                                                    or a list of filenames | np.array (batch imgs inference.

    Returns:
        result (dict): results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # Build the data pipeline
    test_pipeline = Compose(cfg.data.test.pipeline)

    # Prepare data
    if isinstance(imgs, dict):
        data = imgs
        data = test_pipeline(data)
        device = int(str(device).split(":")[-1])
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    elif isinstance(imgs, (str, np.ndarray)):
        # If the input is single image
        data = dict(img=imgs)
        data = test_pipeline(data)
        device = int(str(device).split(":")[-1])
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    else:
        # If the input are batch of images
        batch_data = []
        for img in imgs:
            if isinstance(img, dict):
                data = dict(img_info=img)
            else:
                data = dict(img=img)
            data = test_pipeline(data)
            batch_data.append(data)
        data_collate = collate(batch_data, samples_per_gpu=len(batch_data))
        device = int(str(device).rsplit(':', maxsplit=1)[-1])
        data = scatter(data_collate, [device])[0]

    # Forward inference
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

results = inference_model(model,img)[0]
print(results)
