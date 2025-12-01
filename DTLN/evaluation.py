import copy
import pesq
import torch
import numpy as np
from tqdm import tqdm


def stftLayer(x, blockLen, block_shift):
    frames = x.unfold(-1, blockLen, block_shift)
    return frames
    print(frames.size())
    stft_dat = torch.fft.rfft(frames)
    mag = stft_dat.abs()
    phase = stft_dat.angle()
    return mag, phase


def overlapAddLayer(x, block_shift):
    batch_size, num_frames, frame_length = x.shape
    output_length = block_shift * (num_frames - 1) + frame_length
    output = torch.zeros(batch_size, output_length, device=x.device)
    for i in range(num_frames):
        start = i * block_shift
        end = start + frame_length
        output[:, start:end] += x[:, i]
    return output


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
            input.div_(scale).sub_(zero_point).sub_(128).round_().clamp_(-128, 127)
    else:
        input.div_(scale).sub_(zero_point).round_().clamp_(clamp_min, clamp_max).add_(zero_point).mul_(scale)


"""
Define Floating point 32(FP32) inference
"""
def inference_FP32(model, dataloaders, device):
    model.eval()
    model.to(device)
    dataloader=copy.deepcopy(dataloaders)
    print("Start FP32 inference")
    print("state initialize")
    sum_pesq = 0.0
    count = 0.0
    blockLen = 512
    block_shift = 128

    with torch.no_grad():
        for clean_speech, noisy_speech in tqdm(dataloader):
            state_1 = torch.zeros(1, 2, 128, 2).to(device)
            state_2 = torch.zeros(1, 2, 128, 2).to(device)
            len_orig = len(noisy_speech)
            zero_pad = np.zeros(384)
            noisy_speech_pad = np.concatenate((zero_pad, noisy_speech, zero_pad), axis=0)
            # preprocess
            x = stftLayer(torch.from_numpy(noisy_speech_pad.astype(np.float32)), blockLen, block_shift)
            predicted = None
            for i in range(x.shape[0]):
                inp = x[i,:].reshape(1,1,512).to(device)
                out2, state_1, state_2 = model(inp, state_1, state_2)
                if predicted == None:
                    predicted = out2
                else:
                    predicted = torch.concat((predicted, out2), dim=1)
            # postprocess
            predicted = overlapAddLayer(predicted, block_shift)
            predicted_speech = predicted.squeeze(0)
            predicted_speech = predicted_speech[384:384+len_orig]
            result_pesq = pesq.pesq(ref=clean_speech, deg=predicted_speech.to('cpu').numpy(), fs=16000, mode='nb')
            count = count + 1
            sum_pesq += result_pesq
            print(sum_pesq/count)
        avg_pesq = sum_pesq/count
        print(f"Result fp32 acc is %f" % avg_pesq)
    return avg_pesq


"""
define Fake Quantization (FQ) model inferences
"""
def inference_FQ(model, dataloaders, data_config, device, symm=True, bits=8, calibration=False):
    model.eval()
    model.to(device)
    dataloader=copy.deepcopy(dataloaders)
    print("Start FP32 inference")
    print("state initialize")
    sum_pesq = 0.0
    count = 0.0
    blockLen = 512
    block_shift = 128

    with torch.no_grad():
        for clean_speech, noisy_speech in tqdm(dataloader):
            state_1 = torch.zeros(1, 2, 128, 2).to(device)
            state_2 = torch.zeros(1, 2, 128, 2).to(device)
            len_orig = len(noisy_speech)
            zero_pad = np.zeros(384)
            noisy_speech_pad = np.concatenate((zero_pad, noisy_speech, zero_pad), axis=0)
            # preprocess
            x = stftLayer(torch.from_numpy(noisy_speech_pad.astype(np.float32)), blockLen, block_shift)
            predicted = None
            for i in range(x.shape[0]):
                inp = x[i,:].reshape(1,1,512).to(device)
                state_1.clamp_(data_config['fp32_min'], data_config['fp32_max'])
                out2, state_1, state_2 = model(inp, state_1, state_2)
                if predicted == None:
                    predicted = out2
                else:
                    predicted = torch.concat((predicted, out2), dim=1)
            # postprocess
            predicted = overlapAddLayer(predicted, block_shift)
            predicted_speech = predicted.squeeze(0)
            predicted_speech = predicted_speech[384:384+len_orig]
            result_pesq = pesq.pesq(ref=clean_speech, deg=predicted_speech.to('cpu').numpy(), fs=16000, mode='nb')
            count = count + 1
            if calibration and count==20:
                return 0.0
            sum_pesq += result_pesq
        avg_pesq = sum_pesq/count
        print(f"Result fq acc is %f" % avg_pesq)
    return avg_pesq


"""
Define Hardware(HW) Quantization model inference
"""
def inference_HW(model, dataloaders, data_config, device, symm=True, bits=8, calibration=False):
    model.eval()
    model.to(device)
    dataloader=copy.deepcopy(dataloaders)
    print("Start accuracy estimator inference")
    print("state initialize")
    sum_pesq = 0.0
    count = 0.0
    blockLen = 512
    block_shift = 128

    with torch.no_grad():
        for clean_speech, noisy_speech in tqdm(dataloader):
            state_1 = torch.zeros(1, 2, 128, 2).to(device)
            state_2 = torch.zeros(1, 2, 128, 2).to(device)
            len_orig = len(noisy_speech)
            zero_pad = np.zeros(384)
            noisy_speech_pad = np.concatenate((zero_pad, noisy_speech, zero_pad), axis=0)
            # preprocess
            x = stftLayer(torch.from_numpy(noisy_speech_pad.astype(np.float32)), blockLen, block_shift)
            predicted = None
            for i in range(x.shape[0]):
                inp = x[i,:].reshape(1,1,512).to(device)
                state_1.clamp_(data_config['fp32_min'], data_config['fp32_max'])
                out2, state_1, state_2 = model(inp, state_1, state_2)
                if predicted == None:
                    predicted = out2
                else:
                    predicted = torch.concat((predicted, out2), dim=1)
            # postprocess
            predicted = overlapAddLayer(predicted, block_shift)
            predicted_speech = predicted.squeeze(0)
            predicted_speech = predicted_speech[384:384+len_orig]
            result_pesq = pesq.pesq(ref=clean_speech, deg=predicted_speech.to('cpu').numpy(), fs=16000, mode='nb')
            count = count + 1
            if calibration and count==20:
                return 0.0
            sum_pesq += result_pesq
        avg_pesq = sum_pesq/count
        print(f"Result accuracy estimator acc is %f" % avg_pesq)
    return avg_pesq


def inference_Backend(interpreter, dataloader, data_config, device, symm=True, bits=8):
    assert NotImplementedError


def inference_c(interpreter, dataloader, out_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    output_scale = output_details[0]['quantization_parameters']['scales'][0]
    output_zp = output_details[0]['quantization_parameters']['zero_points'][0]
    output_shape = output_details[0]['shape']
    output_dtype = output_details[0]['dtype']
    dataloader=copy.deepcopy(dataloader)
    print("Start c inference")
    sum_pesq = 0.0
    blockLen = 512
    block_shift = 128
    data_count = 0 

    with torch.no_grad():
        for clean_speech, noisy_speech in tqdm(dataloader):
            len_orig = len(noisy_speech)
            zero_pad = np.zeros(384)
            noisy_speech_pad = np.concatenate((zero_pad, noisy_speech, zero_pad), axis=0)
            # preprocess
            x = stftLayer(torch.from_numpy(noisy_speech_pad.astype(np.float32)), blockLen, block_shift)
            predicted = None
            for i in range(x.shape[0]):
                with open (out_path + "/out_" + str(data_count) + "_" + str(i) + '.bin', 'rb') as fi:
                    res = np.fromfile(fi, output_dtype).reshape(output_shape)
                out2 = output_scale * (res.astype(np.float32) - output_zp)
                out2 = torch.from_numpy(out2)
                if predicted == None:
                    predicted = out2
                else:
                    predicted = torch.concat((predicted, out2), dim=1)
            # postprocess
            predicted = overlapAddLayer(predicted, block_shift)
            predicted_speech = predicted.squeeze(0)
            predicted_speech = predicted_speech[384:384+len_orig]
            result_pesq = pesq.pesq(ref=clean_speech, deg=predicted_speech.to('cpu').numpy(), fs=16000, mode='nb')
            data_count = data_count + 1
            sum_pesq += result_pesq
        avg_pesq = sum_pesq / data_count
        print(f"Result c backend acc is %f" % avg_pesq)
    return avg_pesq
