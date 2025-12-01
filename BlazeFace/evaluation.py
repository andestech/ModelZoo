import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from .dataset import dataset_cfg
from .Model.blazeface.config import cfg_blaze
from .Model.blazeface.models.module.prior_box import PriorBox
from .Model.blazeface.models.module.py_cpu_nms import py_cpu_nms
from .Model.blazeface.utils.box_utils import decode, decode_landm
from .Model.blazeface.evaluator.widerface_evaluate.evaluation import evaluation

now_dir = os.path.dirname(__file__)


"""
Pre_qunat model input tools
"""
def QuantStub(input, min_val=-1.0, max_val=1.0, symm=True, bits=8, isHW=False):
    assert max_val > min_val, "max_val must larger than min_val"
    if symm:
        clamp_min = -((2 ** (bits - 1)))
        clamp_max = 2 ** (bits - 1) - 1
        scale = torch.max(torch.tensor(min_val).abs(), torch.tensor(max_val).abs()).div((2 ** (bits - 1)) - 1)
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
Define Floating point 32(FP32) inference
"""
def inference_FP32(model, dataloader, device):
    model_cfg = dataset_cfg()
    save_folder = './results/'
    testset_folder = model_cfg['testset_folder']
    cfg = cfg_blaze
    model.eval()
    model.to(device)
    print("Start FP32 inference")
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for i, img_name in tqdm(enumerate(dataloader)):
            image_path = testset_folder + img_name
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = np.float32(img_raw)

            # testing scale
            target_size = 640
            max_size = 640
            im_shape = img.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            height, width, _ = im_shape
            image_t = np.empty((im_size_max, im_size_max, 3), dtype=img.dtype)
            image_t[:, :] = (104, 117, 123)
            image_t[0:0 + height, 0:0 + width] = img
            img = cv2.resize(image_t, (max_size, max_size))
            resize = float(target_size) / float(im_size_max)

            im_height, im_width, _ = img.shape
            scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device)
            scale = scale.to(device)
            loc, conf, landms = model(img)
            conf = torch.softmax(conf, -1)
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance']).to(device)
            boxes = boxes.clamp(max=1, min=0.00001)
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            boxes = np.nan_to_num(boxes)
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance']).to(device)
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            inds = np.where(scores > 0.01)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1]
            # order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, 0.5)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            # dets = dets[:args.keep_top_k, :]
            # landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)
            save_name = save_folder + img_name[:-4] + ".txt"
            dirname = os.path.dirname(save_name)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            with open(save_name, "w") as fd:
                bboxs = dets
                file_name = os.path.basename(save_name)[:-4] + "\n"
                bboxs_num = str(len(bboxs)) + "\n"
                fd.write(file_name)
                fd.write(bboxs_num)
                for box in bboxs:
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2]) - int(box[0])
                    h = int(box[3]) - int(box[1])
                    confidence = str(box[4])
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                    fd.write(line)

    acc = evaluation(save_folder, now_dir + "/Model/blazeface/evaluator/widerface_evaluate/ground_truth/")
    print(f"Result fp32 acc is %f" % acc)
    return float(acc) * 100


"""
define Fake Quantization (FQ) model inferences
"""
def inference_FQ(model, dataloader, data_config, device, symm=True, bits=8, calibration=False):
    model_cfg = dataset_cfg()
    save_folder = './results/'
    cfg = cfg_blaze
    model.eval()
    model.to(device)
    print("Start FQ inference")
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for i, img_name in tqdm(enumerate(dataloader)):
            if isinstance(img_name,list):
                img = img_name[0]
                img = img.to(device)
            else:
                testset_folder = model_cfg['testset_folder']
                image_path = testset_folder + img_name
                img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img = np.float32(img_raw)
                # testing scale
                target_size = 640
                max_size = 640
                im_shape = img.shape
                im_size_min = np.min(im_shape[0:2])
                im_size_max = np.max(im_shape[0:2])
                height, width, _ = im_shape
                image_t = np.empty((im_size_max, im_size_max, 3), dtype=img.dtype)
                image_t[:, :] = (104, 117, 123)
                image_t[0:0 + height, 0:0 + width] = img
                img = cv2.resize(image_t, (max_size, max_size))
                resize = float(target_size) / float(im_size_max)
                im_height, im_width, _ = img.shape
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                img -= (104, 117, 123)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0)
                img = img.to(device)
                scale = scale.to(device)
            loc, conf, landms = model(img)
            conf = torch.softmax(conf, -1)
            if calibration:
                continue
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance']).to(device)
            boxes = boxes.clamp(max=1, min=0.00001)
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            boxes = np.nan_to_num(boxes)
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance']).to(device)
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            inds = np.where(scores > 0.01)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1]
            # order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, 0.5)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            # dets = dets[:args.keep_top_k, :]
            # landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)
            save_name = save_folder + img_name[:-4] + ".txt"
            dirname = os.path.dirname(save_name)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            with open(save_name, "w") as fd:
                bboxs = dets
                file_name = os.path.basename(save_name)[:-4] + "\n"
                bboxs_num = str(len(bboxs)) + "\n"
                fd.write(file_name)
                fd.write(bboxs_num)
                for box in bboxs:
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2]) - int(box[0])
                    h = int(box[3]) - int(box[1])
                    confidence = str(box[4])
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                    fd.write(line)

    if calibration:
        print("calibration no map need")
        return 0.0

    acc = evaluation(save_folder, now_dir + "/Model/blazeface/evaluator/widerface_evaluate/ground_truth/")
    print(f"Result fq acc is %f" % acc)
    return float(acc) * 100


"""
Define Hardware(HW) Quantization model inference
"""
def inference_HW(model, dataloader, data_config, device, symm=True, bits=8):
    model_cfg = dataset_cfg()
    save_folder = './results/'
    cfg = cfg_blaze
    model.eval()
    model.to(device)
    print("Start Accuracy estimator inference")
    num_correct = 0
    num_total = 0
    calibration = False
    with torch.no_grad():
        for i, img_name in tqdm(enumerate(dataloader)):
            if isinstance(img_name,list):
                calibration = True
                img = img_name[0]
                img = img.to(device)
            else:
                testset_folder=model_cfg['testset_folder']
                image_path = testset_folder + img_name
                img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img = np.float32(img_raw)
                # testing scale
                target_size = 640
                max_size = 640
                im_shape = img.shape
                im_size_min = np.min(im_shape[0:2])
                im_size_max = np.max(im_shape[0:2])
                height, width, _ = im_shape
                image_t = np.empty((im_size_max, im_size_max, 3), dtype=img.dtype)
                image_t[:, :] = (104, 117, 123)
                image_t[0:0 + height, 0:0 + width] = img
                img = cv2.resize(image_t, (max_size, max_size))
                resize = float(target_size) / float(im_size_max)
                im_height, im_width, _ = img.shape
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                img -= (104, 117, 123)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0)
                img = img.to(device)
                scale = scale.to(device)
            #QuantStub(img,data_config['fp32_min'],data_config['fp32_max'],symm,bits,isHW=False)
            loc, conf, landms = model(img)
            conf = torch.softmax(conf, -1)
            if calibration:
                continue
            priorbox = PriorBox(cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            prior_data = priors.data
            boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance']).to(device)
            boxes = boxes.clamp(max=1, min=0.00001)
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()
            boxes = np.nan_to_num(boxes)
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
            landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance']).to(device)
            scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
            scale1 = scale1.to(device)
            landms = landms * scale1 / resize
            landms = landms.cpu().numpy()

            inds = np.where(scores > 0.01)[0]
            boxes = boxes[inds]
            landms = landms[inds]
            scores = scores[inds]

            # keep top-K before NMS
            order = scores.argsort()[::-1]
            # order = scores.argsort()[::-1][:args.top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, 0.5)
            # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
            dets = dets[keep, :]
            landms = landms[keep]

            # keep top-K faster NMS
            # dets = dets[:args.keep_top_k, :]
            # landms = landms[:args.keep_top_k, :]

            dets = np.concatenate((dets, landms), axis=1)
            save_name = save_folder + img_name[:-4] + ".txt"
            dirname = os.path.dirname(save_name)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            with open(save_name, "w") as fd:
                bboxs = dets
                file_name = os.path.basename(save_name)[:-4] + "\n"
                bboxs_num = str(len(bboxs)) + "\n"
                fd.write(file_name)
                fd.write(bboxs_num)
                for box in bboxs:
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2]) - int(box[0])
                    h = int(box[3]) - int(box[1])
                    confidence = str(box[4])
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                    fd.write(line)

    if calibration:
        print("calibration no map need")
        return 0.0

    acc = evaluation(save_folder, now_dir + "/Model/blazeface/evaluator/widerface_evaluate/ground_truth/")
    print(f"Result accuracy estimator acc is %f" % acc)
    return float(acc)


def inference_Backend(interpreter, dataloader, data_config, device, symm=True, bits=8):
    print("Start inference Backend")
    model_cfg = dataset_cfg()
    save_folder = './results/'
    testset_folder = model_cfg['testset_folder']
    cfg = cfg_blaze
    num_correct = 0
    num_total = 0
    device="cpu"

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    input_dtype = input_details[0]['dtype']
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    output_scale = output_details[0]['quantization_parameters']['scales'][0]
    output_zp = output_details[0]['quantization_parameters']['zero_points'][0]

    # Three output
    for i in range(len(output_details)):
        if output_details[i]["shape"][-1] == 10:
            output_index_landms = i
        elif output_details[i]["shape"][-1] == 4:
            output_index_loc = i
        elif output_details[i]["shape"][-1] == 2:
            output_index_conf = i
        else:
            print("output shape unknown")

    for img_name in tqdm(dataloader):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        target_size = 640
        max_size = 640
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        height, width, _ = im_shape
        image_t = np.empty((im_size_max, im_size_max, 3), dtype=img.dtype)
        image_t[:, :] = (104, 117, 123)
        image_t[0:0 + height, 0:0 + width] = img
        img = cv2.resize(image_t, (max_size, max_size))
        resize = float(target_size) / float(im_size_max)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        img.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)

        input_data = input_dtype(img.numpy().transpose(0, 2, 3, 1))
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        result = interpreter.get_tensor(output_details[output_index_loc]['index'])
        output_scale = output_details[output_index_loc]['quantization_parameters']['scales'][0]
        output_zp = output_details[output_index_loc]['quantization_parameters']['zero_points'][0]
        result = output_scale * (result.astype(np.float64) - output_zp)
        loc = torch.from_numpy(result.transpose(0,1,2))

        result = interpreter.get_tensor(output_details[output_index_conf]['index'])
        output_scale = output_details[output_index_conf]['quantization_parameters']['scales'][0]
        output_zp = output_details[output_index_conf]['quantization_parameters']['zero_points'][0]
        result = output_scale * (result.astype(np.float64) - output_zp)
        conf = torch.from_numpy(result.transpose(0,1,2))

        result = interpreter.get_tensor(output_details[output_index_landms]['index'])
        output_scale = output_details[output_index_landms]['quantization_parameters']['scales'][0]
        output_zp = output_details[output_index_landms]['quantization_parameters']['zero_points'][0]
        result = output_scale * (result.astype(np.float64) - output_zp)
        landms = torch.from_numpy(result.transpose(0,1,2))
        conf = torch.softmax(conf, -1)
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance']).to(device)
        boxes = boxes.clamp(max=1, min=0.00001)
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        boxes = np.nan_to_num(boxes)
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance']).to(device)
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        inds = np.where(scores > 0.01)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.5)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        save_name = save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

    acc = evaluation(save_folder, now_dir + "/Model/blazeface/evaluator/widerface_evaluate/ground_truth/")
    print(f"Result HW acc is %f" % acc)
    return acc * 100


def inference_c(interpreter, dataloader, out_path):
    print("Start inference c")
    model_cfg = dataset_cfg()
    save_folder = './results/'
    testset_folder = model_cfg['testset_folder']
    cfg = cfg_blaze
    num_correct = 0
    num_total = 0
    data_count = 0
    device="cpu"

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    input_dtype = input_details[0]['dtype']
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    output_scale = output_details[0]['quantization_parameters']['scales'][0]
    output_zp = output_details[0]['quantization_parameters']['zero_points'][0]
    output_dtype = output_details[0]["dtype"]

    # Three output
    for i in range(len(output_details)):
        if output_details[i]["shape"][-1] == 10:
            output_index_landms = i
        elif output_details[i]["shape"][-1] == 4:
            output_index_loc = i
        elif output_details[i]["shape"][-1] == 2:
            output_index_conf = i
        else:
            print("output shape unknown")

    for img_name in tqdm(dataloader):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # testing scale
        target_size = 640
        max_size = 640
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        height, width, _ = im_shape
        image_t = np.empty((im_size_max, im_size_max, 3), dtype=img.dtype)
        image_t[:, :] = (104, 117, 123)
        image_t[0:0 + height, 0:0 + width] = img
        img = cv2.resize(image_t, (max_size, max_size))
        resize = float(target_size) / float(im_size_max)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        img.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)

        input_data = input_dtype(img.numpy().transpose(0, 2, 3, 1))

        shape = output_details[output_index_loc]['shape']
        output_dtype = output_details[output_index_loc]['dtype']
        with open (out_path + f"/out_{data_count}_{output_index_loc}.bin", 'rb') as fi:
            result = np.fromfile(fi, output_dtype).reshape(shape)
        output_scale = output_details[output_index_loc]['quantization_parameters']['scales'][0]
        output_zp = output_details[output_index_loc]['quantization_parameters']['zero_points'][0]
        result = output_scale * (result.astype(np.float64) - output_zp)
        loc = torch.from_numpy(result.transpose(0,1,2))

        shape = output_details[output_index_conf]['shape']
        output_dtype = output_details[output_index_conf]['dtype']
        with open (out_path + f"/out_{data_count}_{output_index_conf}.bin", 'rb') as fi:
            result = np.fromfile(fi, output_dtype).reshape(shape)
        output_scale = output_details[output_index_conf]['quantization_parameters']['scales'][0]
        output_zp = output_details[output_index_conf]['quantization_parameters']['zero_points'][0]
        result = output_scale * (result.astype(np.float64) - output_zp)
        conf = torch.from_numpy(result.transpose(0,1,2))

        shape = output_details[output_index_landms]['shape']
        output_dtype = output_details[output_index_landms]['dtype']
        with open (out_path + f"/out_{data_count}_{output_index_landms}.bin", 'rb') as fi:
            result = np.fromfile(fi, output_dtype).reshape(shape)
        output_scale = output_details[output_index_landms]['quantization_parameters']['scales'][0]
        output_zp = output_details[output_index_landms]['quantization_parameters']['zero_points'][0]
        result = output_scale * (result.astype(np.float64) - output_zp)
        landms = torch.from_numpy(result.transpose(0,1,2))
        data_count = data_count + 1
        conf = torch.softmax(conf, -1)
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance']).to(device)
        boxes = boxes.clamp(max=1, min=0.00001)
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        boxes = np.nan_to_num(boxes)
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance']).to(device)
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        inds = np.where(scores > 0.01)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.5)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        save_name = save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

    acc = evaluation(save_folder, now_dir + "/Model/blazeface/evaluator/widerface_evaluate/ground_truth/")
    print(f"Result HW acc is %f" % acc)
    return acc * 100
