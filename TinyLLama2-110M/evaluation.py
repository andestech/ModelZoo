import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer

now_dir=os.path.dirname(__file__)
tokenizer = AutoTokenizer.from_pretrained(f"nickypro/tinyllama-110M")


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
    eval_iters = 5000
    data_pair = dataloader(split='val')
    total_loss= 0.0
    nlls = []
    cont = 0

    with torch.no_grad():
        for count in tqdm(range(eval_iters)):
            input_ids, targets = next(data_pair)
            attention_mask = torch.ones(input_ids.size(-1)).reshape(1,-1).to(device).to(torch.float32)
            position_ids = torch.arange(0, input_ids.size(-1)).reshape(1,-1).to(device).to(torch.float32)
            output = model(input_ids.to(device),attention_mask,position_ids)
            logits = output[0]
            token_ids = torch.argmax(logits, dim=-1).reshape(-1)
            val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.to(device).view(-1), ignore_index=-1)
            total_loss = val_loss+total_loss

    ppl = torch.exp(total_loss / eval_iters)
    print(f"Result FQ ppl is %f" % ppl)
    return ppl


"""
define Fake Quantization (FQ) model inferences
"""
def inference_FQ(model, dataloader, data_config, device, symm=True, bits=8, calibration=False):
    model.eval()
    model.to(device)
    print("Start FQ inference")
    if calibration:
        eval_iters = 1000
    else:
        eval_iters = 5000
    data_pair = dataloader(split='val')
    total_loss= 0.0
    nlls = []
    cont = 0

    with torch.no_grad():
        for count in tqdm(range(eval_iters)):
            input_ids, targets = next(data_pair)
            attention_mask = torch.ones(input_ids.size(-1)).reshape(1,-1).to(device).to(torch.float32)
            position_ids = torch.arange(0, input_ids.size(-1)).reshape(1,-1).to(device).to(torch.float32)
            output = model(input_ids.to(device),attention_mask,position_ids)
            logits = output[0]
            token_ids = torch.argmax(logits, dim=-1).reshape(-1)
            val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.to(device).view(-1), ignore_index=-1)
            total_loss = val_loss+total_loss

    ppl = torch.exp(total_loss / eval_iters)
    print(f"Result FQ ppl is %f" % ppl)
    return ppl


def inference_LLM(models, dataloader, data_config, device, symm=True, bits=8, calibration=False):
    for model in models:
        model.eval()
        model.to(device)
    print("Start LLM FQ inference")
    eval_iters = 20
    data_pair = dataloader(split='val')
    total_loss= 0.0
    nlls = []
    cont = 0

    with torch.no_grad():
        for count in tqdm(range(eval_iters)):
            input_ids, targets = next(data_pair)
            # Prefiil phase
            prefill_ids = input_ids[:,0:128]
            attention_mask = torch.ones(prefill_ids.size(-1)).reshape(1, -1).to(device).to(torch.int32)
            position_ids = torch.arange(0, prefill_ids.size(-1)).reshape(1, -1).to(device).to(torch.int32)
            output = models[0](prefill_ids.to(device), attention_mask,position_ids)
            logits = output[0]
            # Decode phase
            attention_mask = torch.zeros(1025).reshape(1, -1).to(device).to(torch.int32)
            for pos in range(128, 1024):
                kv_cache = output[1:25]
                decode_ids = input_ids[:, pos:pos+1]
                decode_input = (decode_ids.to(device))
                attention_mask[:, -(pos+1):] = 1
                position_ids = torch.tensor(pos).reshape(1, -1).to(device).to(torch.int32)
                input_arg = (decode_input, attention_mask, position_ids) + kv_cache
                output = models[1](*input_arg)
                logits = torch.cat([logits, output[0]], dim=1)
            # PPL calculation
            val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.to(device).view(-1), ignore_index=-1)
            total_loss = val_loss+total_loss
            ppl = torch.exp(total_loss / (count + 1))
            print(f"Result LLN ppl for {count + 1}: %f" % ppl)

    ppl = torch.exp(total_loss / eval_iters)
    print(f"Result LLN ppl is %f" % ppl)
    return ppl


"""
Define Hardware(HW) Quantization model inference
"""
def inference_HW(model, dataloader, data_config, device, symm=True, bits=8, calibration=False):
    model.eval()
    model.to(device)
    print("Start Accuracy estimator inference")
    num_correct = 0
    num_total = 0

    with torch.no_grad():
        for ii, sample in enumerate(dataloader):
            image, label = sample[0].to(device), sample[1].numpy()
            logits = model(image)
            pred = torch.max(logits, 1)[1].cpu().numpy()
            num_correct += np.sum(pred == label)
            num_total += image.shape[0]

    acc = (num_correct / num_total) * 100
    print(f"Result Accuracy estimator acc is %f" % acc)
    return acc


def inference_Backend(interpreter, dataloader, data_config, device, symm=True, bits=8):
    print("Start inference Backend")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_ids_index = 0
    attention_mask_index = 1
    position_ids_index = 2

    input_ids_dtype = input_details[input_ids_index]['dtype']
    input_ids_qmin = np.iinfo(input_ids_dtype).min
    input_ids_qmax = np.iinfo(input_ids_dtype).max

    attention_mask_scale = input_details[attention_mask_index]['quantization_parameters']['scales'][0]
    attention_mask_zp = input_details[attention_mask_index]['quantization_parameters']['zero_points'][0]
    attention_mask_dtype = input_details[attention_mask_index]['dtype']
    attention_mask_qmin = np.iinfo(attention_mask_dtype).min
    attention_mask_qmax = np.iinfo(attention_mask_dtype).max

    position_ids_scale = input_details[position_ids_index]['quantization_parameters']['scales'][0]
    position_ids_zp = input_details[position_ids_index]['quantization_parameters']['zero_points'][0]
    position_ids_dtype = input_details[position_ids_index]['dtype']
    position_ids_qmin = np.iinfo(position_ids_dtype).min
    position_ids_qmax = np.iinfo(position_ids_dtype).max

    output_scale = output_details[0]['quantization_parameters']['scales'][0]
    output_zp = output_details[0]['quantization_parameters']['zero_points'][0]

    data_pair = dataloader(split='val')
    total_loss= 0.0
    eval_iters = 5000

    with torch.no_grad():
        for count in tqdm(range(eval_iters)):
            input_ids, targets = next(data_pair)
            attention_mask = torch.ones(input_ids.size(-1)).reshape(1,-1).to(device).to(torch.float32)
            position_ids = torch.arange(0, input_ids.size(-1)).reshape(1,-1).to(device).to(torch.float32)

            input_ids_data = input_ids.to(device).to(torch.int32)
            input_ids_data.clamp_(input_ids_qmin, input_ids_qmax)
            input_ids_data = input_ids_dtype(input_ids_data.numpy())
            interpreter.set_tensor(input_details[input_ids_index]['index'], input_ids_data)

            attention_mask_data = attention_mask.to(device).to(torch.float32)
            attention_mask_data.div_(attention_mask_scale).round_().add_(attention_mask_zp).clamp_(attention_mask_qmin, attention_mask_qmax)
            attention_mask_data = attention_mask_dtype(attention_mask_data.numpy())
            interpreter.set_tensor(input_details[attention_mask_index]['index'], attention_mask_data)

            position_ids_data = position_ids.to(device).to(torch.float32)
            position_ids_data.div_(position_ids_scale).round_().add_(position_ids_zp).clamp_(position_ids_qmin, position_ids_qmax)
            position_ids_data = position_ids_dtype(position_ids_data.numpy())
            interpreter.set_tensor(input_details[position_ids_index]['index'], position_ids_data)

            interpreter.invoke()
            tflite_out = interpreter.get_tensor(output_details[0]['index'])
            tflite_out_fp32 = output_scale * (tflite_out.astype(np.float32) - output_zp)
            logits = torch.from_numpy(tflite_out_fp32)
            token_ids = torch.argmax(logits, dim=-1).reshape(-1)
            val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.to(device).view(-1), ignore_index=-1)
            total_loss = val_loss + total_loss

    ppl = torch.exp(total_loss / eval_iters)
    print(f"Result Backend ppl is %f" % ppl)
    return ppl


def inference_Backend_LLM(interpreters, dataloader, data_config, device, symm=True, bits=8):
    print("Start Backend LLM inference")
    prefill_interp = interpreters['prefill_128']
    decode_interp  = interpreters['decode']

    prefill_interp.allocate_tensors()
    decode_interp.allocate_tensors()

    p_in  = prefill_interp.get_input_details()
    p_out = prefill_interp.get_output_details()
    d_in  = decode_interp.get_input_details()
    d_out = decode_interp.get_output_details()

    p_ids_dtype = p_in[0]['dtype']
    p_ids_qmin  = np.iinfo(p_ids_dtype).min
    p_ids_qmax  = np.iinfo(p_ids_dtype).max

    p_amask_scale = p_in[1]['quantization_parameters']['scales'][0]
    p_amask_zp    = p_in[1]['quantization_parameters']['zero_points'][0]
    p_amask_dtype = p_in[1]['dtype']
    p_amask_qmin  = np.iinfo(p_amask_dtype).min
    p_amask_qmax  = np.iinfo(p_amask_dtype).max

    p_pos_scale = p_in[2]['quantization_parameters']['scales'][0]
    p_pos_zp    = p_in[2]['quantization_parameters']['zero_points'][0]
    p_pos_dtype = p_in[2]['dtype']
    p_pos_qmin  = np.iinfo(p_pos_dtype).min
    p_pos_qmax  = np.iinfo(p_pos_dtype).max

    d_ids_dtype = d_in[0]['dtype']
    d_ids_qmin  = np.iinfo(d_ids_dtype).min
    d_ids_qmax  = np.iinfo(d_ids_dtype).max

    d_amask_scale = d_in[1]['quantization_parameters']['scales'][0]
    d_amask_zp    = d_in[1]['quantization_parameters']['zero_points'][0]
    d_amask_dtype = d_in[1]['dtype']
    d_amask_qmin  = np.iinfo(d_amask_dtype).min
    d_amask_qmax  = np.iinfo(d_amask_dtype).max

    d_pos_scale = d_in[2]['quantization_parameters']['scales'][0]
    d_pos_zp    = d_in[2]['quantization_parameters']['zero_points'][0]
    d_pos_dtype = d_in[2]['dtype']
    d_pos_qmin  = np.iinfo(d_pos_dtype).min
    d_pos_qmax  = np.iinfo(d_pos_dtype).max

    p_logit_scale = p_out[0]['quantization_parameters']['scales'][0]
    p_logit_zp    = p_out[0]['quantization_parameters']['zero_points'][0]
    d_logit_scale = d_out[0]['quantization_parameters']['scales'][0]
    d_logit_zp    = d_out[0]['quantization_parameters']['zero_points'][0]

    data_pair  = dataloader(split='val')
    eval_iters = 20
    total_loss = 0.0

    with torch.no_grad():
        for count in tqdm(range(eval_iters)):
            input_ids, targets = next(data_pair)

            # Prefill phase
            prefill_ids = input_ids[:, 0:128]

            p_ids_data = prefill_ids.to(torch.int32).numpy()
            p_ids_data = p_ids_data.clip(p_ids_qmin, p_ids_qmax).astype(p_ids_dtype)
            prefill_interp.set_tensor(p_in[0]['index'], p_ids_data)

            p_amask = np.ones((1, 128), dtype=np.float32)
            p_amask = (p_amask / p_amask_scale + p_amask_zp).round().clip(p_amask_qmin, p_amask_qmax).astype(p_amask_dtype)
            prefill_interp.set_tensor(p_in[1]['index'], p_amask)

            p_pos = np.arange(0, 128, dtype=np.float32).reshape(1, 128)
            p_pos = (p_pos / p_pos_scale + p_pos_zp).round().clip(p_pos_qmin, p_pos_qmax).astype(p_pos_dtype)
            prefill_interp.set_tensor(p_in[2]['index'], p_pos)

            prefill_interp.invoke()

            raw = prefill_interp.get_tensor(p_out[0]['index'])
            logits = torch.from_numpy(p_logit_scale * (raw.astype(np.float32) - p_logit_zp))
            kv_cache = [prefill_interp.get_tensor(p_out[i]['index']) for i in range(1, 25)]

            # Decode phase
            attention_mask = np.zeros((1, 1025), dtype=np.float32)

            for pos in range(128, 1024):
                d_ids_data = input_ids[:, pos:pos+1].to(torch.int32).numpy()
                d_ids_data = d_ids_data.clip(d_ids_qmin, d_ids_qmax).astype(d_ids_dtype)
                decode_interp.set_tensor(d_in[0]['index'], d_ids_data)

                attention_mask[:, -(pos+1):] = 1.0
                d_amask = (attention_mask / d_amask_scale + d_amask_zp).round().clip(d_amask_qmin, d_amask_qmax).astype(d_amask_dtype)
                decode_interp.set_tensor(d_in[1]['index'], d_amask)

                d_pos = np.array([[pos]], dtype=np.float32)
                d_pos = (d_pos / d_pos_scale + d_pos_zp).round().clip(d_pos_qmin, d_pos_qmax).astype(d_pos_dtype)
                decode_interp.set_tensor(d_in[2]['index'], d_pos)

                for i, kv in enumerate(kv_cache):
                    decode_interp.set_tensor(d_in[3 + i]['index'], kv)

                decode_interp.invoke()

                raw = decode_interp.get_tensor(d_out[0]['index'])
                step_logits = torch.from_numpy(d_logit_scale * (raw.astype(np.float32) - d_logit_zp))
                logits = torch.cat([logits, step_logits], dim=1)
                kv_cache = [decode_interp.get_tensor(d_out[i]['index']) for i in range(1, 25)]

            # PPL calculation
            val_loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.to('cpu').view(-1), ignore_index=-1)
            total_loss = val_loss + total_loss
            ppl        = torch.exp(total_loss / (count + 1))
            print(f"Result Backend LLM ppl for {count+1}: %f" % ppl)

    ppl = torch.exp(total_loss / eval_iters)
    print(f"Result Backend LLM ppl is %f" % ppl)
    return ppl


def inference_c(interpreter, dataloader, out_path):
    print("Start inference c Backend")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_ids_index = 0
    attention_mask_index = 1
    position_ids_index = 2

    input_ids_dtype = input_details[input_ids_index]['dtype']
    input_ids_qmin = np.iinfo(input_ids_dtype).min
    input_ids_qmax = np.iinfo(input_ids_dtype).max

    attention_mask_scale = input_details[attention_mask_index]['quantization_parameters']['scales'][0]
    attention_mask_zp = input_details[attention_mask_index]['quantization_parameters']['zero_points'][0]
    attention_mask_dtype = input_details[attention_mask_index]['dtype']
    attention_mask_qmin = np.iinfo(attention_mask_dtype).min
    attention_mask_qmax = np.iinfo(attention_mask_dtype).max

    position_ids_scale = input_details[position_ids_index]['quantization_parameters']['scales'][0]
    position_ids_zp = input_details[position_ids_index]['quantization_parameters']['zero_points'][0]
    position_ids_dtype = input_details[position_ids_index]['dtype']
    position_ids_qmin = np.iinfo(position_ids_dtype).min
    position_ids_qmax = np.iinfo(position_ids_dtype).max

    output_scale = output_details[0]['quantization_parameters']['scales'][0]
    output_zp = output_details[0]['quantization_parameters']['zero_points'][0]
    output_dtype = output_details[0]['dtype']
    output_shape = output_details[0]['shape']

    data_pair = dataloader(split='val')
    total_loss= 0.0
    eval_iters = 5000

    with torch.no_grad():
        for count in tqdm(range(eval_iters)):
            input_ids, targets = next(data_pair)
            with open (out_path + "/out_" + str(count) + '.bin', 'rb') as fi:
                res = np.fromfile(fi, output_dtype).reshape(output_shape)
            tflite_out = res
            tflite_out_fp32 = output_scale * (tflite_out.astype(np.float32) - output_zp)
            logits = torch.from_numpy(tflite_out_fp32)
            token_ids = torch.argmax(logits, dim=-1).reshape(-1)
            val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.to('cpu').view(-1), ignore_index=-1)
            total_loss = val_loss + total_loss

    ppl = torch.exp(total_loss / eval_iters)
    print(f"Result c Backend ppl is %f" % ppl)
    return ppl
