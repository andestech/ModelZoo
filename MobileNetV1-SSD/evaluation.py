
# Copyright (C) 2023-2025 Andes Technology Corporation. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import torch
import pathlib
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from .dataset import VOC_classnames
from .Model.voc_utils.data_utils import evaluation
from .Model.voc_utils.data_utils.utils import box_utils
from .Model.SSDs.config import mobilenetv1_ssd_config
from .Model.SSDs.utils import box_utils as box_utils_new
from .Model.SSDs.data_preprocessing import PredictionTransform
from .Model.SSDs.mobilenetv1_ssd import create_mobilenetv1_ssd
from .Model.SSDs.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite_predictor


def model_device(model):
    """Determine the device the model is allocated on."""
    if isinstance(model, nn.DataParallel):
        return model.src_device_obj
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        pass
    return "cpu"


"""
Pre_qunat model input tools
"""
class Post_Net(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.config = mobilenetv1_ssd_config
        self.priors = self.config.priors.to(model_device(net))
        self.net = net

    def forward(self, x):
        confidences, locations = self.net(x)
        confidences = torch.nn.functional.softmax(confidences, dim=2)
        boxes = box_utils_new.convert_locations_to_boxes(
            locations, self.priors, self.config.center_variance, self.config.size_variance
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        return confidences, boxes

def QuantStub(input, min_val=-1.0, max_val=1.0, symm=True, bits=8, isHW=False):
    assert max_val > min_val, "max_val must larger than min_val"
    if symm:
        clamp_min = -((2 ** (bits - 1)))
        clamp_max = 2 ** (bits - 1) - 1
        scale = torch.max(torch.tensor(min_val).abs(), torch.tensor(max_val).abs()).div(( 2** (bits - 1)) - 1)
        zero_point = torch.tensor(0.0)
    else:
        clamp_min = 0
        clamp_max = (2 ** bits) - 1
        scale = (torch.tensor(max_val) - torch.tensor(min_val)).div((2 ** bits) - 1)
        zero_point = torch.tensor(min_val).div(scale).round()
    if isHW:
        if symm:
            input.div_(scale).sub_(zero_point).round_().clamp_(clamp_min, clamp_max)
        else:
            input.div_(scale).sub_(zero_point).sub_(128).round_().clamp_(-128, 127).add_(128).add_(zero_point)
    else:
        input.div_(scale).sub_(zero_point).round_().clamp_(clamp_min, clamp_max).add_(zero_point).mul_(scale)


"""
VOC_use
"""
def eva_result(results,dataset):
    # This should be the part of post_processing, and evaluation
    print("evaluate")
    class_names = VOC_classnames()
    true_case_stat, all_gb_boxes, all_difficult_cases = evaluation.group_annotation_by_class(dataset)
    eval_path = pathlib.Path("Eval_Results")
    eval_path.mkdir(exist_ok=True)
    results = torch.cat(results)

    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = eval_path / "det_test_{0}.txt".format(class_name)
        with open(str(prediction_path), "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )

    aps = []
    print("\n\nAverage Precision Per-class:")

    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = eval_path / "det_test_{0}.txt".format(class_name)
        ap = evaluation.compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            str(prediction_path),
            0.5,
            True
        )
        aps.append(ap)
        print("{0}: {1}".format(class_name, ap))

    print("\nAverage Precision Across All Classes:{0}".format(sum(aps)/len(aps)))
    return sum(aps)/len(aps)


"""
Define Floating point 32(FP32) inference
"""
def inference_FP32(net, voc_data, device):
    os.system('rm -rf Eval_Results')
    net.eval()
    net = net.to(device)
    net = Post_Net(net)
    predictor =  create_mobilenetv1_ssd_lite_predictor(net, nms_method='hard', device=device)
    results = []

    for i in range(len(voc_data)):
        image_raw = voc_data.get_image(i)
        image = predictor.transform(image_raw)
        boxes, labels, probs = predictor.predict([image_raw,image])
        if boxes.numel() == 0 and labels.numel() == 0 and probs.numel() == 0:
            continue
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))

    mAP = eva_result(results,voc_data)
    return mAP * 100


"""
define Fake Quantization (FQ) model inference
"""
def inference_FQ(net, voc_data, data_config, device, symm=True,bits=8,calibration=False):
    os.system('rm -rf Eval_Results')
    net.eval()
    print("Start Evaluation")
    net = net.to(device)
    net=Post_Net(net)
    predictor =  create_mobilenetv1_ssd_lite_predictor(net, nms_method='hard', device=device)
    results = []

    for i in range(len(voc_data)):
        image_raw = voc_data.get_image(i)
        image = predictor.transform(image_raw)
        QuantStub(image, data_config['fp32_min'], data_config['fp32_max'], symm,bits, isHW=False)
        boxes, labels, probs = predictor.predict([image_raw,image])
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))

    if not calibration:
        mAP = eva_result(results,voc_data)
    else:
        mAP= 0.0
    return mAP * 100


"""
Define Hardware(HW) Quantization model inference
"""
def inference_HW(net, voc_data, data_config, device, symm=True,bits=8, calibration=False):
    os.system('rm -rf Eval_Results')
    net.eval()
    print("Start Accuracy estimator Evaluation")
    net = net.to(device)
    net=Post_Net(net)
    predictor =  create_mobilenetv1_ssd_lite_predictor(net, nms_method='hard', device=device)
    results = []

    for i in range(len(voc_data)):
        image_raw = voc_data.get_image(i)
        image = predictor.transform(image_raw)
        boxes, labels, probs = predictor.predict([image_raw,image])
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))

    if not calibration:
        mAP = eva_result(results,voc_data)
    else:
        mAP= 0.0
    return mAP * 100


def inference_Backend(interpreter, dataloader, data_config, device, symm=True, bits=8):
    model_test = create_mobilenetv1_ssd(len(VOC_classnames()), is_test=True)
    model_test.load(os.path.dirname(__file__) + '/Model/SSDs/weights/mobilenet-v1-ssd-mp-0_675.pth')
    model_test.eval()
    priors = model_test.priors
    print("Start inference Backend")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    input_dtype = input_details[0]["dtype"]
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    output_scale = output_details[0]['quantization_parameters']['scales'][0]
    output_zp = output_details[0]['quantization_parameters']['zero_points'][0]
    correct_count = 0
    data_count = 0
    transform = PredictionTransform(300, np.array([127, 127, 127]), 128.0)
    predictor = create_mobilenetv1_ssd_lite_predictor(model_test, nms_method='hard', device=device)
    results = []
    output_index_scores = 0
    j=0
    output_index_boxes = 1

    for i in tqdm(range(len(dataloader))):
        image = dataloader.get_image(i)
        height, width, _ = image.shape
        image = transform(image)
        inputs = image
        if len(inputs.unsqueeze(0).numpy().shape) == 4:
            input_data = inputs.unsqueeze(0)
            input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
            input_data = input_dtype(input_data.numpy().transpose(0, 2, 3, 1))
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            box = interpreter.get_tensor(output_details[output_index_boxes]['index'])
            output_scale = output_details[output_index_boxes]['quantization_parameters']['scales'][0]
            output_zp = output_details[output_index_boxes]['quantization_parameters']['zero_points'][0]
            box_f = output_scale * (box.astype(np.float64) - output_zp)
            box_f = torch.from_numpy(box_f)
            boxes = (box_f, priors.to(device), 0.1, 0.2)
            scores = interpreter.get_tensor(output_details[output_index_scores]['index'])
            output_scale = output_details[output_index_scores]['quantization_parameters']['scales'][0]
            output_zp = output_details[output_index_scores]['quantization_parameters']['zero_points'][0]
            scores_f = output_scale * (scores.astype(np.float64) - output_zp)
            scores_f = torch.from_numpy(scores_f)
            scores_f_softmax = torch.nn.functional.softmax(scores_f, dim=2, _stacklevel=3, dtype=None)
            box_utils.device = device
            boxes = box_utils.convert_locations_to_boxes(*boxes,device)
            boxes = box_utils.center_form_to_corner_form(boxes)
            boxes, labels, probs = predictor.post_processing(boxes, scores_f_softmax, height, width)
            indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
            results.append(torch.cat([
                indexes.reshape(-1, 1),
                labels.reshape(-1, 1).float(),
                probs.reshape(-1, 1),
                boxes.reshape(-1, 4) + 1.0  # matlab's indexes start from 1
            ], dim=1))
            data_count += 1
        else:
            raise Exception("Error shape number")
        j += 1

    mAP = eva_result(results,dataloader)
    print(f"Result HW acc is %f" % mAP)
    return mAP * 100


def inference_c(interpreter, dataloader, out_path):
    model_test = create_mobilenetv1_ssd(len(VOC_classnames()), is_test=True)
    model_test.load(os.path.dirname(__file__) + '/Model/SSDs/weights/mobilenet-v1-ssd-mp-0_675.pth')
    model_test.eval()
    device="cpu"
    priors = model_test.priors
    print("Start inference c")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    output_scale = output_details[0]['quantization_parameters']['scales'][0]
    output_zp = output_details[0]['quantization_parameters']['zero_points'][0]
    correct_count = 0
    data_count = 0
    transform = PredictionTransform(300, np.array([127, 127, 127]), 128.0)
    predictor = create_mobilenetv1_ssd_lite_predictor(model_test, nms_method='hard', device=device)
    results = []
    output_index_scores = 0
    j=0
    output_index_boxes = 1

    for i in tqdm(range(len(dataloader))):
        image = dataloader.get_image(i)
        height, width, _ = image.shape
        image = transform(image)
        inputs = image
        if len(inputs.unsqueeze(0).numpy().shape) == 4:
            shape0 = output_details[output_index_boxes]['shape']
            output_dtype0 = output_details[output_index_boxes]["dtype"]
            with open (out_path + "/out_" + str(data_count) + '_2.bin', 'rb') as fi:
                res0 = np.fromfile(fi,output_dtype0).reshape(shape0)
            box = res0
            output_scale = output_details[output_index_boxes]['quantization_parameters']['scales'][0]
            output_zp = output_details[output_index_boxes]['quantization_parameters']['zero_points'][0]
            box_f = output_scale * (box.astype(np.float64) - output_zp)
            box_f = torch.from_numpy(box_f)
            boxes = (box_f, priors.to(device), 0.1, 0.2)
            shape1 = output_details[output_index_scores]['shape']
            output_dtype1 = output_details[output_index_scores]["dtype"]
            with open (out_path + "/out_" + str(data_count) + '_1.bin', 'rb') as fi:
                res1 = np.fromfile(fi,output_dtype1).reshape(shape1)
            scores = res1
            output_scale = output_details[output_index_scores]['quantization_parameters']['scales'][0]
            output_zp = output_details[output_index_scores]['quantization_parameters']['zero_points'][0]
            scores_f = output_scale * (scores.astype(np.float64) - output_zp)
            scores_f = torch.from_numpy(scores_f)
            scores_f_softmax = torch.nn.functional.softmax(scores_f, dim=2, _stacklevel=3, dtype=None)
            box_utils.device = device
            boxes = box_utils.convert_locations_to_boxes(*boxes,device)
            boxes = box_utils.center_form_to_corner_form(boxes)
            boxes, labels, probs = predictor.post_processing(boxes, scores_f_softmax, height, width)
            indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
            results.append(torch.cat([
                indexes.reshape(-1, 1),
                labels.reshape(-1, 1).float(),
                probs.reshape(-1, 1),
                boxes.reshape(-1, 4) + 1.0  # matlab's indexes start from 1
            ], dim=1))
            data_count += 1
        else:
            raise Exception("Error shape number")
        j += 1

    mAP = eva_result(results,dataloader)
    print(f"Result HW acc is %f" % mAP)
    return mAP * 100
