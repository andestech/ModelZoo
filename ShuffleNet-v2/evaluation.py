import torch
import numpy as np
from tqdm import tqdm

"""
Pre_qunat model input tools
"""
def QuantStub(input,min_val=-1.0,max_val=1.0,symm=True,bits=8,isHW=False):
    assert max_val>min_val,"max_val must larger than min_val"
    if symm:
        clamp_min=-((2**(bits-1)))#for bits=8 -128
        clamp_max=2**(bits-1)-1   #for bits=8 127
        scale=torch.max(torch.tensor(min_val).abs(),torch.tensor(max_val).abs()).div((2**(bits-1))-1)
        zero_point=torch.tensor(0.0)
    else:
        clamp_min=0
        clamp_max=(2**bits)-1
        scale=(torch.tensor(max_val)-torch.tensor(min_val)).div((2**bits)-1)
        zero_point=torch.tensor(min_val).div(scale).round()
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
    with torch.no_grad():
        for ii, sample in enumerate(dataloader):
            image, label = sample[0].to(device), sample[1].numpy()
            logits = model(image)
            pred = torch.max(logits, 1)[1].cpu().numpy()

            num_correct += np.sum(pred == label)
            num_total += image.shape[0]
            # print(num_correct, num_total, num_correct/num_total)
    acc = (num_correct / num_total)*100
    print(f"Result fp32 acc is %f" % acc)
    return acc

"""
define Fake Quantization (FQ) model inferences
"""

def inference_FQ(model, dataloader, data_config, device, symm=True, bits=8,calibration=False):
    model.eval()
    model.to(device)
    print("Start FQ inference")
    num_correct = 0
    num_total = 0
    scale = 0.02078740157480315
    zero_point=0
    with torch.no_grad():
        for ii, sample in enumerate(dataloader):
            image, label = sample[0].to(device), sample[1].numpy()
            #image.div_(scale).sub_(zero_point).round_().clamp_(-128, 127).add_(zero_point).mul_(scale)
            #image.div_(scale).round_().clamp_(-128, 127).mul_(scale)
            QuantStub(image,data_config['fp32_min'],data_config['fp32_max'],symm,bits,isHW=False)#input, dynamic_range min/max, isHW(Hardware or Fakequant) 
            logits = model(image)

            pred = torch.max(logits, 1)[1].cpu().numpy()

            num_correct += np.sum(pred == label)
            num_total += image.shape[0]
            # print(num_correct, num_total, num_correct/num_total)
    acc = (num_correct / num_total)*100
    print(f"Result FQ acc is %f" % acc)
    return acc

"""
Define Hardware(HW) Quantization model inference
"""

def inference_HW(model, dataloader, data_config, device, symm=True, bits=8):
    model.eval()
    model.to(device)
    print("Start Accuracy estimator inference")
    num_correct = 0
    num_total = 0
    scale = 0.02078740157480315
    zero_point=0
    with torch.no_grad():
        for ii, sample in enumerate(dataloader):
            image, label = sample[0].to(device), sample[1].numpy()
            #image.div_(scale).sub_(zero_point).round_().clamp_(-128, 127).add_(zero_point).mul_(scale)
            #image.div_(scale).round_().clamp_(-128, 127).mul_(scale)
            #QuantStub(image,data_config['fp32_min'],data_config['fp32_max'],symm,bits,isHW=False)#input, dynamic_range min/max, isHW(Hardware or Fakequant) 
            logits = model(image)

            pred = torch.max(logits, 1)[1].cpu().numpy()

            num_correct += np.sum(pred == label)
            num_total += image.shape[0]
            # print(num_correct, num_total, num_correct/num_total)
    acc = (num_correct / num_total)*100
    print(f"Result Accuracy estimator acc is %f" % acc)
    return acc

def compare(fq, tf, detail):
    torch.set_printoptions(precision=10)
    cos = torch.nn.CosineSimilarity(dim=0)

    zp = detail['quantization_parameters']['zero_points']
    scale = detail['quantization_parameters']['scales']

    tf_fp = torch.from_numpy(np.float32((np.float32(tf) - np.float32(zp)) * scale))

    print(cos(tf_fp.to(torch.float64).cpu().flatten(), fq.to(torch.float64).cpu().flatten()))
    return

def inference_Backend(interpreter, dataloader, data_config, device, symm=True, bits=8):
    print("Start inference Backend")
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

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.numpy()
            for i in range(inputs.shape[0]):
                if len(inputs[i].unsqueeze(0).numpy().shape) == 4:
                    input_data = inputs[i].unsqueeze(0)
                    input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
                    input_data = input_dtype(input_data.numpy().transpose(0, 2, 3, 1))
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    tflite_out = interpreter.get_tensor(output_details[0]['index'])
                    if int(tflite_out.argmax()) == labels[i].item():
                        correct_count += 1
                    data_count += 1
                else:
                    raise Exception("Error shape number")
    acc = (correct_count / data_count) * 100

    print(f"Result Backend acc is %f" % acc)
    return acc

def inference_c(interpreter, dataloader, out_path):
    print("Start inference c")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    output_scale = output_details[0]['quantization_parameters']['scales'][0]
    output_zp = output_details[0]['quantization_parameters']['zero_points'][0]
    output_shape = output_details[0]['shape']
    output_dtype = output_details[0]["dtype"]
    correct_count = 0
    data_count = 0
    for inputs, labels in tqdm(dataloader):
        for i in range(inputs.shape[0]):
            if len(inputs[i].unsqueeze(0).numpy().shape) == 4:
                with open (out_path + "/out_" + str(data_count) + '.bin', 'rb') as fi:
                    res = np.fromfile(fi, output_dtype).reshape(output_shape)
                tflite_out = res
                if int(tflite_out.argmax()) == labels[i].item():
                    correct_count += 1
                data_count += 1
            else:
                raise Exception("Error shape number")
    acc = (correct_count / data_count) * 100

    print(f"Result C Backend acc is %f" % acc)
    return acc

"""
Define Quantization model inference for metrics
"""
def forward_one(model, dataloaders, device):
    for inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)
        output = None
        del output

def forward_one_Q(model, dataloader, data_config, device, symm=True, bits=8):
    for inputs, labels in dataloader:
        image = inputs.to(device)
        QuantStub(image,data_config['fp32_min'],data_config['fp32_max'],symm,bits,isHW=False)
        labels = labels.to(device)
        output = model(image)
        output = None
        del output
