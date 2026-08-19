import torch
import evaluate
import numpy as np
from tqdm import tqdm
from .dataset import dataset_feature
from transformers import EvalPrediction
from .utils import postprocess_qa_predictions


"""
Pre_qunat model input tools
"""
def QuantStub(input, min_val=-1.0, max_val=1.0, symm=True, bits=8, isHW=False):
    assert max_val > min_val, "max_val must larger than min_val"
    if symm:
        clamp_min = -((2 ** (bits - 1)))  # for bits=8 -128
        clamp_max = 2 ** (bits - 1) - 1   # for bits=8 127
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


def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=True,
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
        output_dir=None,
        log_level=20,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if True:
        formatted_predictions = [
            {"id": str(k), "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": str(k), "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": str(ex["id"]), "answers": ex["answers"]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


"""
Define Floating point 32(FP32) inference
"""
def inference_FP32(model, dataloader, device):
    print("Start FP32 inference")
    model.eval()
    model.to(device)
    num_correct = 0
    num_total = 0
    example,feature = dataset_feature()
    metric = evaluate.load("squad_v2")
    output_start = []
    output_end = []
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_ids = sample['input_ids'].to(device)
            attention_mask = sample['attention_mask'].to(device).to(torch.float32)
            token_type_ids = sample['token_type_ids'].to(device)
            logits = model(input_ids, attention_mask, token_type_ids)
            pred_start, pred_end = logits
            output_start.append(pred_start)
            output_end.append(pred_end)
    output_start = torch.cat(output_start, dim=0)
    output_end = torch.cat(output_end, dim=0)
    prediction = torch.stack([output_start, output_end], dim=0).detach().cpu().numpy()
    result = post_processing_function(example, feature, prediction, stage="eval")
    result = metric.compute(predictions=result.predictions, references=result.label_ids)
    print(result['f1'])
    return float(result['f1'])


"""
define Fake Quantization (FQ) model inferences
"""
def inference_FQ(model, dataloader, data_config, device, symm=True, bits=8, calibration=False):
    print("Start FQ inference")
    model.eval()
    model.to(device)
    num_correct = 0
    num_total = 0
    example,feature =dataset_feature()
    metric = evaluate.load("squad_v2")
    output_start = []
    output_end = []
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_ids = sample['input_ids'].to(device)
            attention_mask = sample['attention_mask'].to(device).to(torch.float32)
            token_type_ids = sample['token_type_ids'].to(device)
            logits = model(input_ids, attention_mask, token_type_ids)
            pred_start, pred_end = logits
            output_start.append(pred_start)
            output_end.append(pred_end)
    if calibration:
        return 0.0
    output_start = torch.cat(output_start,dim=0)
    output_end = torch.cat(output_end,dim=0)
    prediction = torch.stack([output_start, output_end], dim=0).detach().cpu().numpy()
    result = post_processing_function(example, feature, prediction, stage="eval")
    result = metric.compute(predictions=result.predictions, references=result.label_ids)
    print(result['f1'])
    return float(result['f1'])


"""
Define Hardware(HW) Quantization model inference
"""
def inference_HW(model, dataloader, data_config, device, symm=True, bits=8):
    print("Start Accuracy estimator inference")
    model.eval()
    model.to(device)
    num_correct = 0
    num_total = 0
    scale = 0.02078740157480315
    zero_point = 0
    with torch.no_grad():
        for ii, sample in enumerate(dataloader):
            image, label = sample[0].to(device), sample[1].numpy()
            logits = model(image)
            pred = torch.max(logits, 1)[1].cpu().numpy()
            num_correct += np.sum(pred == label)
            num_total += image.shape[0]
            # print(num_correct, num_total, num_correct/num_total)
    acc = (num_correct / num_total) * 100
    print(f"Result Accuracy estimator acc is %f" % acc)
    return acc


def inference_Backend(interpreter, dataloader, data_config, device, symm=True, bits=8):
    print("Start inference Backend")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_ids_idx = 0
    attention_mask_idx = 1
    token_type_ids_idx = 2
    pred_start_idx = 0
    pred_end_idx = 1

    input_ids_dtype = input_details[input_ids_idx]['dtype']
    assert input_ids_dtype is np.int32, "Input id dtype must be int32 for BERT-like models, but got %s" % input_ids_dtype

    attention_mask_dtype = input_details[attention_mask_idx]['dtype']
    attention_mask_scale = input_details[attention_mask_idx]['quantization_parameters']['scales'][0]
    attention_mask_zp = input_details[attention_mask_idx]['quantization_parameters']['zero_points'][0]
    attention_mask_qmin = np.iinfo(attention_mask_dtype).min
    attention_mask_qmax = np.iinfo(attention_mask_dtype).max

    token_type_ids_dtype = input_details[token_type_ids_idx]['dtype']
    assert token_type_ids_dtype is np.int32, "Input id dtype must be int32 for BERT-like models, but got %s" % token_type_ids_dtype

    pred_start_scale = output_details[pred_start_idx]['quantization_parameters']['scales'][0]
    pred_start_zp = output_details[pred_start_idx]['quantization_parameters']['zero_points'][0]
    pred_end_scale = output_details[pred_end_idx]['quantization_parameters']['scales'][0]
    pred_end_zp = output_details[pred_end_idx]['quantization_parameters']['zero_points'][0]

    example, feature = dataset_feature()
    metric = evaluate.load("squad_v2")
    output_start = []
    output_end = []
    with torch.no_grad():
        for sample in tqdm(dataloader):
            for i in range(sample['input_ids'].shape[0]):
                input_ids = sample['input_ids'][i].unsqueeze(0).to(device)
                attention_mask = sample['attention_mask'][i].unsqueeze(0).to(device)
                token_type_ids = sample['token_type_ids'][i].unsqueeze(0).to(device)

                input_ids_data = input_ids.cpu().numpy().astype(input_ids_dtype)
                attention_mask = attention_mask.to(float).div_(attention_mask_scale).round_().add_(attention_mask_zp).clamp_(attention_mask_qmin, attention_mask_qmax)
                attention_mask_data = attention_mask.cpu().numpy().astype(attention_mask_dtype)
                token_type_ids_data = token_type_ids.cpu().numpy().astype(token_type_ids_dtype)

                interpreter.set_tensor(input_details[input_ids_idx]['index'], input_ids_data)
                interpreter.set_tensor(input_details[attention_mask_idx]['index'], attention_mask_data)
                interpreter.set_tensor(input_details[token_type_ids_idx]['index'], token_type_ids_data)
                interpreter.invoke()

                pred_start = interpreter.get_tensor(output_details[pred_start_idx]['index'])
                pred_start = pred_start_scale * (pred_start.astype(np.float32) - pred_start_zp)
                pred_start = torch.from_numpy(pred_start)
                pred_end = interpreter.get_tensor(output_details[pred_end_idx]['index'])
                pred_end = pred_end_scale * (pred_end.astype(np.float32) - pred_end_zp)
                pred_end = torch.from_numpy(pred_end)
                output_start.append(pred_start)
                output_end.append(pred_end)

    output_start = torch.cat(output_start,dim=0)
    output_end = torch.cat(output_end,dim=0)
    prediction = torch.stack([output_start,output_end], dim=0).detach().cpu().numpy()
    result = post_processing_function(example, feature, prediction, stage="eval")
    result = metric.compute(predictions=result.predictions, references=result.label_ids)
    print(f"Result Backend acc is %f" % result['f1'])
    return float(result['f1'])


def inference_c(interpreter, dataloader, out_path):
    print("Start inference c")
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()

    pred_start_idx = 0
    pred_end_idx = 1

    pred_start_scale = output_details[pred_start_idx]['quantization_parameters']['scales'][0]
    pred_start_zp = output_details[pred_start_idx]['quantization_parameters']['zero_points'][0]
    pred_start_dtype = output_details[pred_start_idx]['dtype']
    pred_start_shape = output_details[pred_start_idx]['shape']

    pred_end_scale = output_details[pred_end_idx]['quantization_parameters']['scales'][0]
    pred_end_zp = output_details[pred_end_idx]['quantization_parameters']['zero_points'][0]
    pred_end_dtype = output_details[pred_end_idx]['dtype']
    pred_end_shape = output_details[pred_end_idx]['shape']

    example, feature = dataset_feature()
    metric = evaluate.load("squad_v2")
    output_start = []
    output_end = []
    data_count = 0
    with torch.no_grad():
        for sample in tqdm(dataloader):
            for i in range(sample['input_ids'].shape[0]):
                with open (out_path + "/out_" + str(data_count) + '_0.bin', 'rb') as fi:
                    res = np.fromfile(fi, pred_start_dtype).reshape(pred_start_shape)
                pred_start = res
                pred_start = pred_start_scale * (pred_start.astype(np.float32) - pred_start_zp)
                pred_start = torch.from_numpy(pred_start)

                with open (out_path + "/out_" + str(data_count) + '_1.bin', 'rb') as fi:
                    res = np.fromfile(fi, pred_end_dtype).reshape(pred_end_shape)
                pred_end = res
                pred_end = pred_end_scale * (pred_end.astype(np.float32) - pred_end_zp)
                pred_end = torch.from_numpy(pred_end)

                output_start.append(pred_start)
                output_end.append(pred_end)
                data_count += 1

    output_start = torch.cat(output_start, dim=0)
    output_end = torch.cat(output_end, dim=0)
    prediction = torch.stack([output_start, output_end], dim=0).detach().cpu().numpy()
    result = post_processing_function(example, feature, prediction, stage="eval")
    result = metric.compute(predictions=result.predictions, references=result.label_ids)
    print(f"Result C Backend acc is %f" % result['f1'])
    return float(result['f1'])
