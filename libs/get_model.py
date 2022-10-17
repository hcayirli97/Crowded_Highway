from models.common import DetectMultiBackend

def get_model(weights,device,data,imgsz):

    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    model.eval()
    stride, names, pt = model.stride, model.names, model.pt

    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))
    return model
