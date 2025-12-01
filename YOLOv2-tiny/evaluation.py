import torch
import numpy as np
import pathlib
from .utils import decode_v2, evaluate_map_voc, detect, detect_tflite
from .voc import VOC_CLASS
import cv2
from tqdm import tqdm


"""
Pre_qunat model input tools
"""
def QuantStub(input, min_val=-1.0, max_val=1.0, symm=True, bits=8, isHW=False):
    assert max_val > min_val, "max_val must larger than min_val"
    if symm:
        clamp_min = -((2 ** (bits - 1)))  # for bits=8 -128
        clamp_max = 2 ** (bits - 1) - 1  # for bits=8 127
        scale = torch.max(torch.tensor(min_val).abs(), torch.tensor(max_val).abs()).div((2 ** (bits - 1)) - 1)
        zero_point = torch.tensor(0.0)
    else:
        clamp_min = 0
        clamp_max = (2**bits) - 1
        scale = (torch.tensor(max_val) - torch.tensor(min_val)).div((2**bits) - 1)
        zero_point = torch.tensor(min_val).div(scale).round()
    if isHW:
        if symm:
            input.div_(scale).sub_(zero_point).round_().clamp_(clamp_min, clamp_max)
        else:
            input.div_(scale).sub_(zero_point).sub_(128).round_().clamp_(-128, 127).add_(128).add_(zero_point)
    else:
        input.div_(scale).sub_(zero_point).round_().clamp_(clamp_min, clamp_max).add_(zero_point).mul_(scale)


"""
Define Floating point 32(FP32) inference
"""
def inference_FP32(net, dataloader, device):
    import os

    net.eval()
    decode = decode_v2
    model = net.to(device)
    with torch.no_grad():
        a, b, c, d, e, f, g, h = [], [], [], [], [], [], [], []
        for id in tqdm(range(len(dataloader.dataset))):
            id = torch.tensor(id).to(device)
            input, label = dataloader.dataset[id]

            boxes_detected, classes_detected, probs_detected = detect(
                input.to(device), model, decode, 0.5, num_classes=20, prob_threshold=0.01
            )
            target_bboxes = label[0].to(device)
            target_classes = label[1].to(device)
            target_difficults = label[2].to(device)
            for box_detected, class_detected, prob_detected in zip(boxes_detected, classes_detected, probs_detected):
                a.append(box_detected)
                b.append(class_detected)
                c.append(prob_detected)
                d.append(id)
            for target_bbox, target_class, target_difficult in zip(target_bboxes, target_classes, target_difficults):
                e.append(target_bbox)
                f.append(target_class)
                g.append(target_difficult)
                h.append(id)
        if len(a) != 0:
            aps = evaluate_map_voc(
                torch.stack(a),
                torch.tensor(b),
                torch.tensor(c),
                torch.tensor(d),
                torch.stack(e),
                torch.tensor(f),
                torch.tensor(g),
                torch.tensor(h),
                VOC_CLASS,
            )
    mAP = aps.mean().item()
    print("now mAP is" + str(mAP))
    return mAP * 100


"""
define Fake Quantization (FQ) model inference
"""
def inference_FQ(net, dataloader, data_config, device, symm=True, bits=8, calibration=False):
    import os

    net.eval()
    decode = decode_v2
    model = net.to(device)
    with torch.no_grad():
        a, b, c, d, e, f, g, h = [], [], [], [], [], [], [], []
        for id in tqdm(range(len(dataloader.dataset))):
            id = torch.tensor(id).to(device)
            input, label = dataloader.dataset[id]
            boxes_detected, classes_detected, probs_detected = detect(
                input.to(device), model, decode, 0.5, num_classes=20, prob_threshold=0.01
            )
            target_bboxes = label[0].to(device)
            target_classes = label[1].to(device)
            target_difficults = label[2].to(device)
            for box_detected, class_detected, prob_detected in zip(boxes_detected, classes_detected, probs_detected):
                a.append(box_detected)
                b.append(class_detected)
                c.append(prob_detected)
                d.append(id)
            for target_bbox, target_class, target_difficult in zip(target_bboxes, target_classes, target_difficults):
                e.append(target_bbox)
                f.append(target_class)
                g.append(target_difficult)
                h.append(id)
        if calibration:
            return 0.0
        if len(a) != 0:
            aps = evaluate_map_voc(
                    torch.stack(a),
                    torch.tensor(b),
                    torch.tensor(c),
                    torch.tensor(d),
                    torch.stack(e),
                    torch.tensor(f),
                    torch.tensor(g),
                    torch.tensor(h),
                    VOC_CLASS,
                )
        mAP = aps.mean().item()

    print("now mAP is" + str(mAP))
    return mAP * 100


"""
Define Hardware(HW) Quantization model inference
"""

def inference_HW(net, dataloader, data_config, device, symm=True, bits=8, calibration=False):
    import os

    net.eval()
    decode = decode_v2
    model = net.to(device)
    with torch.no_grad():
        a, b, c, d, e, f, g, h = [], [], [], [], [], [], [], []
        for id in tqdm(range(len(dataloader.dataset))):
            id = torch.tensor(id).to(device)
            input, label = dataloader.dataset[id]
            #QuantStub(input, data_config["fp32_min"], data_config["fp32_max"], symm, bits, isHW=False)
            boxes_detected, classes_detected, probs_detected = detect(
                input.to(device), model, decode, 0.5, num_classes=20, prob_threshold=0.01
            )
            target_bboxes = label[0].to(device)
            target_classes = label[1].to(device)
            target_difficults = label[2].to(device)
            for box_detected, class_detected, prob_detected in zip(boxes_detected, classes_detected, probs_detected):
                a.append(box_detected)
                b.append(class_detected)
                c.append(prob_detected)
                d.append(id)
            for target_bbox, target_class, target_difficult in zip(target_bboxes, target_classes, target_difficults):
                e.append(target_bbox)
                f.append(target_class)
                g.append(target_difficult)
                h.append(id)
        if calibration:
            return 0.0
        if len(a) != 0:
            aps = evaluate_map_voc(
                    torch.stack(a),
                    torch.tensor(b),
                    torch.tensor(c),
                    torch.tensor(d),
                    torch.stack(e),
                    torch.tensor(f),
                    torch.tensor(g),
                    torch.tensor(h),
                    VOC_CLASS,
                )
        mAP = aps.mean().item()

    print("now mAP is" + str(mAP))
    return mAP * 100

def inference_Backend(interpreter, dataloader, data_config, device, symm=True, bits=8):
    print("Start inference Backend")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]["quantization_parameters"]["scales"][0]
    input_zp = input_details[0]["quantization_parameters"]["zero_points"][0]
    input_dtype = input_details[0]['dtype']
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    output_scale = output_details[0]["quantization_parameters"]["scales"][0]
    output_zp = output_details[0]["quantization_parameters"]["zero_points"][0]
    results = []
    a, b, c, d, e, f, g, h = [], [], [], [], [], [], [], []
    decode = decode_v2
    correct_count = 0
    data_count = 0
    for id in tqdm(range(len(dataloader.dataset))):
        id = torch.tensor(id).to(device)
        input, label = dataloader.dataset[id]

        if len(input.numpy().shape) == 3:
            input_data = input.unsqueeze(0)
            input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
            input_data = input_dtype(input_data.numpy().transpose(0, 2, 3, 1))
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            box = interpreter.get_tensor(output_details[0]["index"])
            box_f = output_scale * (box.astype(np.float64) - output_zp)
            box_f = torch.from_numpy(box_f)
            boxes_detected, classes_detected, probs_detected = detect_tflite(
                box_f, decode, 0.5, num_classes=20, prob_threshold=0.01
            )
            target_bboxes = label[0].to(device)
            target_classes = label[1].to(device)
            target_difficults = label[2].to(device)
            for box_detected, class_detected, prob_detected in zip(boxes_detected, classes_detected, probs_detected):
                a.append(box_detected)
                b.append(class_detected)
                c.append(prob_detected)
                d.append(id)
            for target_bbox, target_class, target_difficult in zip(target_bboxes, target_classes, target_difficults):
                e.append(target_bbox)
                f.append(target_class)
                g.append(target_difficult)
                h.append(id)
    try:
        if len(a) != 0:
            aps = evaluate_map_voc(
                torch.stack(a),
                torch.tensor(b),
                torch.tensor(c),
                torch.tensor(d),
                torch.stack(e),
                torch.tensor(f),
                torch.tensor(g),
                torch.tensor(h),
                VOC_CLASS,
            )
        mAP = aps.mean().item()
    except:
        mAP = 0.0
    print("now mAP is" + str(mAP))

    print(f"Result HW map is %f" % mAP)
    return mAP * 100

def inference_c(interpreter, dataloader, out_path):
    print("Start inference c")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]["quantization_parameters"]["scales"][0]
    input_zp = input_details[0]["quantization_parameters"]["zero_points"][0]
    input_dtype = input_details[0]['dtype']
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    output_scale = output_details[0]["quantization_parameters"]["scales"][0]
    output_zp = output_details[0]["quantization_parameters"]["zero_points"][0]
    shape = output_details[0]['shape']
    output_dtype = output_details[0]["dtype"]
    results = []
    a, b, c, d, e, f, g, h = [], [], [], [], [], [], [], []
    decode = decode_v2
    correct_count = 0
    data_count = 0
    device="cpu"

    for id in tqdm(range(len(dataloader.dataset))):
        id = torch.tensor(id).to(device)
        input, label = dataloader.dataset[id]

        if len(input.numpy().shape) == 3:
            with open (out_path + "/out_" +str(data_count) + '.bin', 'rb') as fi:
                res = np.fromfile(fi,output_dtype).reshape(shape)
                data_count=data_count+1
            box = res
            box_f = output_scale * (box.astype(np.float64) - output_zp)
            box_f = torch.from_numpy(box_f)
            boxes_detected, classes_detected, probs_detected = detect_tflite(
                box_f, decode, 0.5, num_classes=20, prob_threshold=0.01
            )
            target_bboxes = label[0].to(device)
            target_classes = label[1].to(device)
            target_difficults = label[2].to(device)
            for box_detected, class_detected, prob_detected in zip(boxes_detected, classes_detected, probs_detected):
                a.append(box_detected)
                b.append(class_detected)
                c.append(prob_detected)
                d.append(id)
            for target_bbox, target_class, target_difficult in zip(target_bboxes, target_classes, target_difficults):
                e.append(target_bbox)
                f.append(target_class)
                g.append(target_difficult)
                h.append(id)
    try:
        if len(a) != 0:
            aps = evaluate_map_voc(
                torch.stack(a),
                torch.tensor(b),
                torch.tensor(c),
                torch.tensor(d),
                torch.stack(e),
                torch.tensor(f),
                torch.tensor(g),
                torch.tensor(h),
                VOC_CLASS,
            )
        mAP = aps.mean().item()
    except:
        mAP = 0.0
    print("now mAP is" + str(mAP))

    print(f"Result HW map is %f" % mAP)
    return mAP * 100


def forward_one(net, dataloader, device):
    import os

    net.eval()
    decode = decode_v2
    model = net.to(device)
    with torch.no_grad():
        a, b, c, d, e, f, g, h = [], [], [], [], [], [], [], []
        for id in tqdm(range(len(dataloader.dataset))):
            id = torch.tensor(id).to(device)
            input, label = dataloader.dataset[id]

            boxes_detected, classes_detected, probs_detected = detect(
                input.to(device), model, decode, 0.5, num_classes=20, prob_threshold=0.01
            )
            break

def forward_one_Q(net, dataloader, data_config, device, symm=True, bits=8):
    import os

    net.eval()
    decode = decode_v2
    model = net.to(device)
    with torch.no_grad():
        a, b, c, d, e, f, g, h = [], [], [], [], [], [], [], []
        for id in tqdm(range(len(dataloader.dataset))):
            id = torch.tensor(id).to(device)
            input, label = dataloader.dataset[id]
            QuantStub(input, data_config["fp32_min"], data_config["fp32_max"], symm, bits, isHW=False)
            boxes_detected, classes_detected, probs_detected = detect(
                input.to(device), model, decode, 0.5, num_classes=20, prob_threshold=0.01
            )
            break
