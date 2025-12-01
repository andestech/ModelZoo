import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize


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
    model.eval()
    model.to(device)
    print("Start FP32 inference")
    num_correct = 0
    num_total = 0
    embs = []
    issame_list = dataloader.dataset.issame_list
    test_issame = np.array(issame_list).astype("bool")

    with torch.no_grad():
        for ii, sample in tqdm(enumerate(dataloader)):
            image, label = sample[0].to(device), sample[1].numpy()
            image = image.permute(0, 3, 1, 2)
            logits = model(image)
            embs.extend(logits.detach().cpu().numpy())

        embs = np.array(embs)
        embs = normalize(embs)
        embs_a = embs[::2]
        embs_b = embs[1::2]
        dists = (embs_a * embs_b).sum(1)
        tt = np.sort(dists[test_issame[: dists.shape[0]]])
        ff = np.sort(dists[np.logical_not(test_issame[:dists.shape[0]])])
        t_steps = int(0.1 * ff.shape[0])
        acc_count = np.array([(tt > vv).sum() + (ff <= vv).sum() for vv in ff[-t_steps:]])
        acc_max_indx = np.argmax(acc_count)
        acc_max = acc_count[acc_max_indx] / dists.shape[0]
        acc_thresh = ff[acc_max_indx - t_steps]
        acc = acc_max * 100

    print(f"Result fp32 acc is %f" % acc)
    return acc


"""
define Fake Quantization (FQ) model inferences
"""
def inference_FQ(model, dataloader, data_config, device, symm=True, bits=8, calibration=False):
    model.eval()
    model.to(device)
    print(model)
    print("Start FQ inference")
    embs = []

    if not calibration:
        issame_list = dataloader.dataset.issame_list
        test_issame = np.array(issame_list).astype("bool")

    with torch.no_grad():
        for ii, sample in tqdm(enumerate(dataloader)):
            image, label = sample[0].to(device), sample[1].numpy()
            image = image.permute(0, 3, 1, 2)
            QuantStub(image,data_config['fp32_min'], data_config['fp32_max'], symm,bits, isHW=False)
            logits = model(image)
            embs.extend(logits.detach().cpu().numpy())

        if calibration:
            print("calibration")
            return 0.0

        embs = np.array(embs)
        embs = normalize(embs)
        embs_a = embs[::2]
        embs_b = embs[1::2]
        dists = (embs_a * embs_b).sum(1)
        tt = np.sort(dists[test_issame[:dists.shape[0]]])
        ff = np.sort(dists[np.logical_not(test_issame[:dists.shape[0]])])
        t_steps = int(0.1 * ff.shape[0])
        acc_count = np.array([(tt > vv).sum() + (ff <= vv).sum() for vv in ff[-t_steps:]])
        acc_max_indx = np.argmax(acc_count)
        acc_max = acc_count[acc_max_indx] / dists.shape[0]
        acc_thresh = ff[acc_max_indx - t_steps]
        acc = acc_max * 100

    print(f"Result FQ acc is %f" % acc)
    return acc


"""
Define Hardware(HW) Quantization model inference
"""
def inference_HW(model, dataloader, data_config, device, symm=True, bits=8, calibration=False):
    model.eval()
    model.to(device)
    print(model)
    print("Start accuracy estimator inference")
    embs = []

    if not calibration:
        issame_list = dataloader.dataset.issame_list
        test_issame = np.array(issame_list).astype("bool")

    with torch.no_grad():
        for ii, sample in tqdm(enumerate(dataloader)):
            image, label = sample[0].to(device), sample[1].numpy()
            image = image.permute(0, 3, 1, 2)
            logits = model(image)
            embs.extend(logits.detach().cpu().numpy())

        if calibration:
            print("calibration")
            return 0.0

        embs = np.array(embs)
        embs = normalize(embs)
        embs_a = embs[::2]
        embs_b = embs[1::2]
        dists = (embs_a * embs_b).sum(1)
        tt = np.sort(dists[test_issame[: dists.shape[0]]])
        ff = np.sort(dists[np.logical_not(test_issame[:dists.shape[0]])])
        t_steps = int(0.1 * ff.shape[0])
        acc_count = np.array([(tt > vv).sum() + (ff <= vv).sum() for vv in ff[-t_steps:]])
        acc_max_indx = np.argmax(acc_count)
        acc_max = acc_count[acc_max_indx] / dists.shape[0]
        acc_thresh = ff[acc_max_indx - t_steps]
        acc = acc_max * 100

    print(f"Result accuracy estimator acc is %f" % acc)
    return acc


def inference_Backend(interpreter, dataloader, data_config, device, symm=True, bits=8):
    print("Start inference Backend")
    issame_list = dataloader.dataset.issame_list
    test_issame = np.array(issame_list).astype("bool")
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
    correct_count = 0
    data_count = 0
    embs=[]

    for sample in tqdm(dataloader):
        image, label = sample[0].to(device), sample[1].numpy()
        input_data = image.permute(0, 1, 2, 3)
        input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
        for batch_idx in range(input_data.size(0)):
            input_data_sub = input_dtype(input_data[batch_idx].unsqueeze(0).numpy())
            interpreter.set_tensor(input_details[0]['index'], input_data_sub)
            interpreter.invoke()
            tflite_out = interpreter.get_tensor(output_details[0]['index'])
            embs.extend(output_scale * (tflite_out.astype(np.float32) - output_zp))

    embs = np.array(embs)
    embs = normalize(embs)
    embs_a = embs[::2]
    embs_b = embs[1::2]
    dists = (embs_a * embs_b).sum(1)
    tt = np.sort(dists[test_issame[: dists.shape[0]]])
    ff = np.sort(dists[np.logical_not(test_issame[:dists.shape[0]])])
    t_steps = int(0.1 * ff.shape[0])
    acc_count = np.array([(tt > vv).sum() + (ff <= vv).sum() for vv in ff[-t_steps:]])
    acc_max_indx = np.argmax(acc_count)
    acc_max = acc_count[acc_max_indx] / dists.shape[0]
    acc_thresh = ff[acc_max_indx - t_steps]
    acc = acc_max * 100

    print(f"Result Backend acc is %f" % acc)
    return acc


def inference_c(interpreter, dataloader, out_path):
    print("Start inference c Backend")
    issame_list = dataloader.dataset.issame_list
    test_issame = np.array(issame_list).astype("bool")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    output_scale = output_details[0]['quantization_parameters']['scales'][0]
    output_zp = output_details[0]['quantization_parameters']['zero_points'][0]
    output_dtype = output_details[0]['dtype']
    shape = output_details[0]['shape']
    data_count = 0
    embs=[]

    for inputs, labels in tqdm(dataloader):
        for i in range(inputs.shape[0]):
            if len(inputs[i].unsqueeze(0).numpy().shape) == 4:
                with open (out_path + "/out_" + str(data_count) + '.bin', 'rb') as fi:
                    res = np.fromfile(fi, output_dtype).reshape(shape)
                tflite_out = res
            embs.extend(output_scale * (tflite_out.astype(np.float32) - output_zp))
            data_count += 1

    embs = np.array(embs)
    embs = normalize(embs)
    embs_a = embs[::2]
    embs_b = embs[1::2]
    dists = (embs_a * embs_b).sum(1)
    tt = np.sort(dists[test_issame[: dists.shape[0]]])
    ff = np.sort(dists[np.logical_not(test_issame[:dists.shape[0]])])
    t_steps = int(0.1 * ff.shape[0])
    acc_count = np.array([(tt > vv).sum() + (ff <= vv).sum() for vv in ff[-t_steps:]])
    acc_max_indx = np.argmax(acc_count)
    acc_max = acc_count[acc_max_indx] / dists.shape[0]
    acc_thresh = ff[acc_max_indx - t_steps]
    acc = acc_max * 100

    print(f"Result c Backend acc is %f" % acc)
    return acc
