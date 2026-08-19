import os
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, default_data_collator


"""
Loading model_cfg.yaml in the current directory
"""
now_dir = os.path.dirname(__file__)
with open(now_dir + "/model_cfg.yaml", "r") as f:
    input_yaml = yaml.load(f, Loader=yaml.FullLoader)

"""
Template for return data pair
"""
HuggingFace_model_path = now_dir + '/Model/output_squadv2/'
raw_datasets = load_dataset(
    "squad_v2",
    None,
    cache_dir=input_yaml['dataset_path'],
    token=None,
    trust_remote_code=False
)
tokenizer = AutoTokenizer.from_pretrained(
    HuggingFace_model_path,
    cache_dir=input_yaml['dataset_path'],
    use_fast=True,
    revision='main',
    token=None,
    trust_remote_code=False,
)

pad_on_right = tokenizer.padding_side == "right"
max_seq_length = 384
doc_stride = 128
column_names = raw_datasets["train"].column_names
question_column_name = "question"
context_column_name = "context"
answer_column_name = "answers"


"""
Dataset value max/min in FP32
"""
def dataset_cfg():
    return input_yaml


def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []
    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]
    return tokenized_examples


def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        if tokenizer.cls_token_id in input_ids:
            cls_index = input_ids.index(tokenizer.cls_token_id)
        elif tokenizer.bos_token_id in input_ids:
            cls_index = input_ids.index(tokenizer.bos_token_id)
        else:
            cls_index = 0
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    return tokenized_examples


def return_dataset():
    eval_examples = raw_datasets["validation"]
    train_examples = raw_datasets["train"]
    train_dataset = train_examples.map(
        prepare_train_features,
        batched=True,
        num_proc=0,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )
    eval_dataset = eval_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=0,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on validation dataset",
    )
    eval_dataset = eval_dataset.remove_columns(['offset_mapping', 'example_id'])
    dataloader_params = {
        "batch_size": input_yaml['batch_size'],
        "collate_fn": default_data_collator,
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
    }
    tra_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_params)
    val_dataloader = DataLoader(eval_dataset, **dataloader_params)
    dataloader_params = {
        "batch_size": 1,
        "collate_fn": default_data_collator,
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
    }
    cos_dataloader = DataLoader(eval_dataset, **dataloader_params)
    return tra_dataloader, val_dataloader, val_dataloader, cos_dataloader


def dataset_feature():
    eval_examples = raw_datasets["validation"]
    eval_feature = eval_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=0,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on validation dataset",
    )
    return eval_examples, eval_feature


def prepare_testbin(interpreter, dataloader, save_path="./output/"):
    print("Start generate test_bin")
    device = "cpu"
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

    data_count = 0
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

                input_ids_data.tofile(save_path + "/test_" + str(data_count) + "_0.bin")
                attention_mask_data.tofile(save_path + "/test_" + str(data_count) + "_1.bin")
                token_type_ids_data.tofile(save_path + "/test_" + str(data_count) + "_2.bin")
                data_count += 1
