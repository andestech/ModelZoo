import torch
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pathlib
import os
import cv2
import yaml
from .utils import recover_pad_output
from .anchor import decode_tf, prior_box_tf
from .widerface_eval import evaluation


now_dir = os.path.dirname(__file__)
with open(now_dir + "/model_cfg.yaml", "r") as f:
    input_yaml = yaml.load(f, Loader=yaml.FullLoader)
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


def inference_FP32(model, dataloader, device):
    model.eval()
    model.to(device)
    print("Start FP32 inference")
    for sample in tqdm(dataloader):
        img, pad_params, img_path = sample[0].to(device), sample[1], sample[2][0]
        img = img.div(127.5).sub(1)
        bbox_regressions, landm_regressions, classifications = model(img)
        boxes = bbox_regressions.detach().cpu().numpy()
        landm = landm_regressions.detach().cpu().numpy()
        classifies = classifications.detach().cpu().numpy()
        preds = tf.concat(  # [bboxes, landms, landms_valid, conf]
            [
                boxes[0],
                landm[0],
                tf.ones_like(classifies[0, :, 0][..., tf.newaxis]),
                classifies[0, :, 1][..., tf.newaxis],
            ],
            1,
        )
        priors = prior_box_tf(
            input_yaml["image_size"], input_yaml["min_sizes"], input_yaml["steps"], input_yaml["clip"]
        )
        decode_preds = decode_tf(preds, priors, input_yaml["variances"])
        selected_indices = tf.image.non_max_suppression(
            boxes=decode_preds[:, :4],
            scores=decode_preds[:, -1],
            max_output_size=tf.shape(decode_preds)[0],
            iou_threshold=0.4,
            score_threshold=0.02,
        )
        out = tf.gather(decode_preds, selected_indices).numpy()
        outputs = recover_pad_output(out, pad_params)
        img_name = os.path.basename(img_path)
        sub_dir = os.path.basename(os.path.dirname(img_path))
        save_name = os.path.join("./widerface_evaluate/widerface_txt/", sub_dir, img_name.replace(".jpg", ".txt"))
        pathlib.Path(os.path.join("./widerface_evaluate/widerface_txt/", sub_dir)).mkdir(parents=True, exist_ok=True)
        img_height_raw, img_width_raw, _ = cv2.imread(img_path, cv2.IMREAD_COLOR).shape
        with open(save_name, "w") as file:
            bboxs = outputs[:, :4]
            confs = outputs[:, -1]
            file_name = img_name + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            file.write(file_name)
            file.write(bboxs_num)
            for box, conf in zip(bboxs, confs):
                x = int(box[0] * img_width_raw)
                y = int(box[1] * img_height_raw)
                w = int(box[2] * img_width_raw) - int(box[0] * img_width_raw)
                h = int(box[3] * img_height_raw) - int(box[1] * img_height_raw)
                confidence = str(conf)
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                file.write(line)
    mAP = evaluation("./widerface_evaluate/widerface_txt/", os.path.join(now_dir, "./ground_truth/"))
    result = float(mAP[0])
    return result * 100


"""
define Fake Quantization (FQ) model inferences
"""


def inference_FQ(model, dataloader, data_config, device, symm=True, bits=8, calibration=False):
    model.eval()
    model.to(device)
    print("Start FQ inference")
    num_correct = 0
    num_total = 0
    scale = 0.02078740157480315
    zero_point = 0
    print(model)
    with torch.no_grad():
        for sample in tqdm(dataloader):
            img, pad_params, img_path = sample[0].to(device), sample[1], sample[2][0]
            img = img.div(127.5).sub(1)  # preprocessing cut by frontend.
            QuantStub(
                img, data_config["fp32_min"], data_config["fp32_max"], symm, bits, isHW=False
            )  # input, dynamic_range min/max, isHW(Hardware or Fakequant)
            bbox_regressions, landm_regressions, classifications = model(img)
            if calibration:
                continue
            boxes = bbox_regressions.cpu().numpy()
            landm = landm_regressions.cpu().numpy()
            classifies = classifications.cpu().numpy()
            preds = tf.concat(  # [bboxes, landms, landms_valid, conf]
                [
                    boxes[0],
                    landm[0],
                    tf.ones_like(classifies[0, :, 0][..., tf.newaxis]),
                    classifies[0, :, 1][..., tf.newaxis],
                ],
                1,
            )
            priors = prior_box_tf(
                input_yaml["image_size"], input_yaml["min_sizes"], input_yaml["steps"], input_yaml["clip"]
            )
            decode_preds = decode_tf(preds, priors, input_yaml["variances"])
            selected_indices = tf.image.non_max_suppression(
                boxes=decode_preds[:, :4],
                scores=decode_preds[:, -1],
                max_output_size=tf.shape(decode_preds)[0],
                iou_threshold=0.4,
                score_threshold=0.02,
            )
            out = tf.gather(decode_preds, selected_indices).numpy()
            outputs = recover_pad_output(out, pad_params)
            img_name = os.path.basename(img_path)
            sub_dir = os.path.basename(os.path.dirname(img_path))
            save_name = os.path.join("./widerface_evaluate/widerface_txt/", sub_dir, img_name.replace(".jpg", ".txt"))
            pathlib.Path(os.path.join("./widerface_evaluate/widerface_txt/", sub_dir)).mkdir(
                parents=True, exist_ok=True
            )
            img_height_raw, img_width_raw, _ = cv2.imread(img_path, cv2.IMREAD_COLOR).shape
            with open(save_name, "w") as file:
                bboxs = outputs[:, :4]
                confs = outputs[:, -1]
                file_name = img_name + "\n"
                bboxs_num = str(len(bboxs)) + "\n"
                file.write(file_name)
                file.write(bboxs_num)
                for box, conf in zip(bboxs, confs):
                    x = int(box[0] * img_width_raw)
                    y = int(box[1] * img_height_raw)
                    w = int(box[2] * img_width_raw) - int(box[0] * img_width_raw)
                    h = int(box[3] * img_height_raw) - int(box[1] * img_height_raw)
                    confidence = str(conf)
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                    file.write(line)
        if not calibration:
            mAP = evaluation("./widerface_evaluate/widerface_txt/", os.path.join(now_dir, "./ground_truth/"))
            result = float(mAP[0])
        else:
            result = 0.0
    return result


"""
Define Hardware(HW) Quantization model inference
"""


def inference_HW(model, dataloader, data_config, device, symm=True, bits=8, calibration=False):
    model.eval()
    model.to(device)
    print("Start Accuracy estimator inference")
    num_correct = 0
    num_total = 0
    print(model)
    with torch.no_grad():
        for sample in tqdm(dataloader):
            img, pad_params, img_path = sample[0].to(device), sample[1], sample[2][0]
            img = img.div(127.5).sub(1)  # preprocessing cut by frontend.
            bbox_regressions, landm_regressions, classifications = model(img)
            boxes = bbox_regressions.cpu().numpy()
            landm = landm_regressions.cpu().numpy()
            classifies = classifications.cpu().numpy()
            preds = tf.concat(  # [bboxes, landms, landms_valid, conf]
                [
                    boxes[0],
                    landm[0],
                    tf.ones_like(classifies[0, :, 0][..., tf.newaxis]),
                    classifies[0, :, 1][..., tf.newaxis],
                ],
                1,
            )
            priors = prior_box_tf(
                input_yaml["image_size"], input_yaml["min_sizes"], input_yaml["steps"], input_yaml["clip"]
            )
            decode_preds = decode_tf(preds, priors, input_yaml["variances"])
            selected_indices = tf.image.non_max_suppression(
                boxes=decode_preds[:, :4],
                scores=decode_preds[:, -1],
                max_output_size=tf.shape(decode_preds)[0],
                iou_threshold=0.4,
                score_threshold=0.02,
            )
            out = tf.gather(decode_preds, selected_indices).numpy()
            outputs = recover_pad_output(out, pad_params)
            img_name = os.path.basename(img_path)
            sub_dir = os.path.basename(os.path.dirname(img_path))
            save_name = os.path.join("./widerface_evaluate/widerface_txt/", sub_dir, img_name.replace(".jpg", ".txt"))
            pathlib.Path(os.path.join("./widerface_evaluate/widerface_txt/", sub_dir)).mkdir(
                parents=True, exist_ok=True
            )
            img_height_raw, img_width_raw, _ = cv2.imread(img_path, cv2.IMREAD_COLOR).shape
            with open(save_name, "w") as file:
                bboxs = outputs[:, :4]
                confs = outputs[:, -1]
                file_name = img_name + "\n"
                bboxs_num = str(len(bboxs)) + "\n"
                file.write(file_name)
                file.write(bboxs_num)
                for box, conf in zip(bboxs, confs):
                    x = int(box[0] * img_width_raw)
                    y = int(box[1] * img_height_raw)
                    w = int(box[2] * img_width_raw) - int(box[0] * img_width_raw)
                    h = int(box[3] * img_height_raw) - int(box[1] * img_height_raw)
                    confidence = str(conf)
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                    file.write(line)
        mAP = evaluation("./widerface_evaluate/widerface_txt/", os.path.join(now_dir, "./ground_truth/"))
        result = float(mAP[0])
    return result


def inference_Backend(interpreter, dataloader, data_config, device, symm=True, bits=8):
    print("Start inference Backend")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]['dtype']
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    input_scale = input_details[0]["quantization_parameters"]["scales"][0]
    input_zp = input_details[0]["quantization_parameters"]["zero_points"][0]
    # Three output
    for i in range(len(output_details)):
        if output_details[i]["shape"][-1] == 10:
            lamda_regression_idx = i
        elif output_details[i]["shape"][-1] == 4:
            boxes_regression_idx = i
        elif output_details[i]["shape"][-1] == 2:
            classification_idx = i
        else:
            print("output shape unknown")
    boxes_regression_scale = output_details[boxes_regression_idx]["quantization_parameters"]["scales"][0]
    boxes_regression_zp = output_details[boxes_regression_idx]["quantization_parameters"]["zero_points"][0]
    boxes_regression_shape = output_details[boxes_regression_idx]["shape"]

    lamda_regression_scale = output_details[lamda_regression_idx]["quantization_parameters"]["scales"][0]
    lamda_regression_zp = output_details[lamda_regression_idx]["quantization_parameters"]["zero_points"][0]
    lamda_regression_shape = output_details[lamda_regression_idx]["shape"]

    classification_scale = output_details[classification_idx]["quantization_parameters"]["scales"][0]
    classification_zp = output_details[classification_idx]["quantization_parameters"]["zero_points"][0]
    classification_shape = output_details[classification_idx]["shape"]
    correct_count = 0
    data_count = 0
    device = "cpu"
    for sample in tqdm(dataloader):
        img, pad_params, img_path = sample[0].to(device), sample[1], sample[2][0]
        input_data = img.div(127.5).sub(1)
        input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
        input_data = input_dtype(input_data.numpy().transpose(0, 2, 3, 1))
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        boxes_regression_out = interpreter.get_tensor(output_details[boxes_regression_idx]["index"])
        lamda_regression_out = interpreter.get_tensor(output_details[lamda_regression_idx]["index"])
        classification_out = interpreter.get_tensor(output_details[classification_idx]["index"])

        boxes_regression_out = boxes_regression_scale * (boxes_regression_out.astype(np.float32) - boxes_regression_zp)
        lamda_regression_out = lamda_regression_scale * (lamda_regression_out.astype(np.float32) - lamda_regression_zp)
        classification_out = classification_scale * (classification_out.astype(np.float32) - classification_zp)
        preds = tf.concat(  # [bboxes, landms, landms_valid, conf]
            [
                boxes_regression_out[0],
                lamda_regression_out[0],
                tf.ones_like(classification_out[0, :, 0][..., tf.newaxis]),
                classification_out[0, :, 1][..., tf.newaxis],
            ],
            1,
        )
        priors = prior_box_tf(
            input_yaml["image_size"], input_yaml["min_sizes"], input_yaml["steps"], input_yaml["clip"]
        )
        decode_preds = decode_tf(preds, priors, input_yaml["variances"])
        selected_indices = tf.image.non_max_suppression(
            boxes=decode_preds[:, :4],
            scores=decode_preds[:, -1],
            max_output_size=tf.shape(decode_preds)[0],
            iou_threshold=0.4,
            score_threshold=0.02,
        )
        out = tf.gather(decode_preds, selected_indices).numpy()
        outputs = recover_pad_output(out, pad_params)
        img_name = os.path.basename(img_path)
        sub_dir = os.path.basename(os.path.dirname(img_path))
        save_name = os.path.join(
            "./widerface_evaluate/widerface_backend_txt/", sub_dir, img_name.replace(".jpg", ".txt")
        )
        pathlib.Path(os.path.join("./widerface_evaluate/widerface_backend_txt/", sub_dir)).mkdir(
            parents=True, exist_ok=True
        )
        img_height_raw, img_width_raw, _ = cv2.imread(img_path, cv2.IMREAD_COLOR).shape
        with open(save_name, "w") as file:
            bboxs = outputs[:, :4]
            confs = outputs[:, -1]
            file_name = img_name + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            file.write(file_name)
            file.write(bboxs_num)
            for box, conf in zip(bboxs, confs):
                x = int(box[0] * img_width_raw)
                y = int(box[1] * img_height_raw)
                w = int(box[2] * img_width_raw) - int(box[0] * img_width_raw)
                h = int(box[3] * img_height_raw) - int(box[1] * img_height_raw)
                confidence = str(conf)
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                file.write(line)
    print(f"Result of Backend is")
    mAP = evaluation("./widerface_evaluate/widerface_backend_txt/", os.path.join(now_dir, "./ground_truth/"))

    return float(mAP[0])


def inference_c(interpreter, dataloader, out_path):
    print("Start inference c Backend")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]["quantization_parameters"]["scales"][0]
    input_zp = input_details[0]["quantization_parameters"]["zero_points"][0]
    # Three output
    for i in range(len(output_details)):
        if output_details[i]["shape"][-1] == 10:
            lamda_regression_idx = i
        elif output_details[i]["shape"][-1] == 4:
            boxes_regression_idx = i
        elif output_details[i]["shape"][-1] == 2:
            classification_idx = i
        else:
            print("output shape unknown")
    boxes_regression_scale = output_details[boxes_regression_idx]["quantization_parameters"]["scales"][0]
    boxes_regression_zp = output_details[boxes_regression_idx]["quantization_parameters"]["zero_points"][0]
    boxes_regression_shape = output_details[boxes_regression_idx]["shape"]
    boxes_regression_dtype = output_details[boxes_regression_idx]["dtype"]

    lamda_regression_scale = output_details[lamda_regression_idx]["quantization_parameters"]["scales"][0]
    lamda_regression_zp = output_details[lamda_regression_idx]["quantization_parameters"]["zero_points"][0]
    lamda_regression_shape = output_details[lamda_regression_idx]["shape"]
    lamda_regression_dtype = output_details[lamda_regression_idx]["dtype"]

    classification_scale = output_details[classification_idx]["quantization_parameters"]["scales"][0]
    classification_zp = output_details[classification_idx]["quantization_parameters"]["zero_points"][0]
    classification_shape = output_details[classification_idx]["shape"]
    classification_dtype = output_details[classification_idx]["dtype"]

    data_count = 0
    device = "cpu"
    for sample in tqdm(dataloader):
        img, pad_params, img_path = sample[0].to(device), sample[1], sample[2][0]
        with open(out_path + "/out_" + str(data_count) + "_1.bin", "rb") as fi:
            lamda_regression_out = np.fromfile(fi, lamda_regression_dtype).reshape(lamda_regression_shape)
        with open(out_path + "/out_" + str(data_count) + "_2.bin", "rb") as fi:
            classification_out = np.fromfile(fi, classification_dtype).reshape(classification_shape)
        with open(out_path + "/out_" + str(data_count) + "_0.bin", "rb") as fi:
            boxes_regression_out = np.fromfile(fi, boxes_regression_dtype).reshape(boxes_regression_shape)
        boxes_regression_out = boxes_regression_scale * (boxes_regression_out.astype(np.float32) - boxes_regression_zp)
        lamda_regression_out = lamda_regression_scale * (lamda_regression_out.astype(np.float32) - lamda_regression_zp)
        classification_out = classification_scale * (classification_out.astype(np.float32) - classification_zp)
        preds = tf.concat(  # [bboxes, landms, landms_valid, conf]
            [
                boxes_regression_out[0],
                lamda_regression_out[0],
                tf.ones_like(classification_out[0, :, 0][..., tf.newaxis]),
                classification_out[0, :, 1][..., tf.newaxis],
            ],
            1,
        )
        priors = prior_box_tf(
            input_yaml["image_size"], input_yaml["min_sizes"], input_yaml["steps"], input_yaml["clip"]
        )
        decode_preds = decode_tf(preds, priors, input_yaml["variances"])
        selected_indices = tf.image.non_max_suppression(
            boxes=decode_preds[:, :4],
            scores=decode_preds[:, -1],
            max_output_size=tf.shape(decode_preds)[0],
            iou_threshold=0.4,
            score_threshold=0.02,
        )
        out = tf.gather(decode_preds, selected_indices).numpy()
        outputs = recover_pad_output(out, pad_params)
        img_name = os.path.basename(img_path)
        sub_dir = os.path.basename(os.path.dirname(img_path))
        save_name = os.path.join(
            "./widerface_evaluate/widerface_backend_txt/", sub_dir, img_name.replace(".jpg", ".txt")
        )
        pathlib.Path(os.path.join("./widerface_evaluate/widerface_backend_txt/", sub_dir)).mkdir(
            parents=True, exist_ok=True
        )
        img_height_raw, img_width_raw, _ = cv2.imread(img_path, cv2.IMREAD_COLOR).shape
        with open(save_name, "w") as file:
            bboxs = outputs[:, :4]
            confs = outputs[:, -1]
            file_name = img_name + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            file.write(file_name)
            file.write(bboxs_num)
            for box, conf in zip(bboxs, confs):
                x = int(box[0] * img_width_raw)
                y = int(box[1] * img_height_raw)
                w = int(box[2] * img_width_raw) - int(box[0] * img_width_raw)
                h = int(box[3] * img_height_raw) - int(box[1] * img_height_raw)
                confidence = str(conf)
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                file.write(line)
        data_count = data_count + 1

    print(f"Result of C Backend is")
    mAP = evaluation("./widerface_evaluate/widerface_backend_txt/", os.path.join(now_dir, "./ground_truth/"))

    return float(mAP[0])


def forward_one(model, dataloader, device):
    model.eval()
    model.to(device)
    print("Start forward one fp32 inference")
    num_correct = 0
    num_total = 0
    scale = 0.02078740157480315
    zero_point = 0
    with torch.no_grad():
        for ii, sample in enumerate(dataloader):
            image, label = sample[0].to(device), sample[1].numpy()
            logits = model(image)
            break


def forward_one_Q(model, dataloader, data_config, device, symm=True, bits=8):
    model.eval()
    model.to(device)
    print("Start forward one FQ inference")
    num_correct = 0
    num_total = 0
    scale = 0.02078740157480315
    zero_point = 0
    with torch.no_grad():
        for ii, sample in enumerate(dataloader):
            image, label = sample[0].to(device), sample[1].numpy()
            QuantStub(
                image, data_config["fp32_min"], data_config["fp32_max"], symm, bits, isHW=False
            )  # input, dynamic_range min/max, isHW(Hardware or Fakequant)
            logits = model(image)
            break
