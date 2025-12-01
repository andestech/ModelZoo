import torch
import numpy as np
from tqdm import tqdm
from .rnnoise_pre_processing import RNNoisePreProcess
import librosa
import pesq
import copy

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
    vad_gru_state = torch.zeros((1, 24)).to(device)
    noise_gru_state = torch.zeros((1, 48)).to(device)
    denoise_gru_state = torch.zeros((1, 96)).to(device)
    sum_pesq = 0.0
    final_denoised_audio = []
    count=0.0

    with torch.no_grad():
        for clean_speech, noisy_speech in dataloader:
            final_denoised_audio=[]
            preprocess = RNNoisePreProcess(training=False)
            num_samples = len(noisy_speech) // preprocess.FRAME_SIZE
            for i in range(num_samples):
                audio_window = noisy_speech[i * RNNoisePreProcess.FRAME_SIZE:
                                        i * RNNoisePreProcess.FRAME_SIZE + RNNoisePreProcess.FRAME_SIZE]
                silence, features, X, P, Ex, Ep, Exp = preprocess.process_frame(audio_window)
                features = np.expand_dims(features, (0, 1)).astype(np.float32)
                if not silence:
                    input_data = torch.from_numpy(features).to(device)
                    denoise_gru_state, denoise_output, noise_gru_state, vad_gru_state, vad_out = model(input_data,vad_gru_state,noise_gru_state,denoise_gru_state)
                    vad_gru_state = torch.squeeze(vad_gru_state, axis=1)
                    noise_gru_state = torch.squeeze(noise_gru_state, axis=1)
                    denoise_gru_state = torch.squeeze(denoise_gru_state, axis=1)

                    denoise_output = torch.squeeze(denoise_output)
                    denoise_output_np = denoise_output.to("cpu").detach().numpy()
                    denoised_audio_tmp = preprocess.post_process(silence, denoise_output_np, X, P, Ex, Ep, Exp)
                    denoised_audio_tmp = np.rint(denoised_audio_tmp)
                    final_denoised_audio.append(denoised_audio_tmp)
                else:
                    final_denoised_audio.append(np.zeros([preprocess.FRAME_SIZE], dtype=np.int16))
            denoised_audio = np.concatenate(final_denoised_audio, axis=0)
            resample_clean = librosa.resample(clean_speech, orig_sr=48000, target_sr=16000)
            resample_denoised = librosa.resample(denoised_audio, orig_sr=48000, target_sr=16000)
            sum_pesq += pesq.pesq(ref=resample_clean, deg=resample_denoised, fs=16000, mode='wb')
            count = count+1
            print("number of count: "+str(count))
            print(sum_pesq/count)
        avg_pesq=sum_pesq/count
        print(avg_pesq)
    print(f"Result fp32 acc is %f" % avg_pesq)
    return avg_pesq

"""
define Fake Quantization (FQ) model inferences
"""
def inference_FQ(model, dataloaders, data_config, device, symm=True, bits=8, calibration=False):
    model.eval()
    model.to(device)
    print("Start FQ inference")
    dataloader=copy.deepcopy(dataloaders)
    print("state initialize")
    vad_gru_state = torch.zeros((1, 24)).to(device)
    noise_gru_state = torch.zeros((1, 48)).to(device)
    denoise_gru_state = torch.zeros((1, 96)).to(device)
    sum_pesq = 0.0
    final_denoised_audio = []
    num_file=0
    num_file_cal=0

    with torch.no_grad():
        for clean_speech, noisy_speech in tqdm(dataloader):
            if calibration:
                if num_file % 8 == 0:
                    num_file_cal = num_file_cal + 1
                    #every 8 file go forward function, otherwise pass
                    pass
                else:
                    num_file = num_file +1
                    continue
            final_denoised_audio=[]
            preprocess = RNNoisePreProcess(training=False)
            num_samples = len(noisy_speech) // preprocess.FRAME_SIZE
            for i in range(num_samples):
                audio_window = noisy_speech[i * RNNoisePreProcess.FRAME_SIZE:
                                        i * RNNoisePreProcess.FRAME_SIZE + RNNoisePreProcess.FRAME_SIZE]
                silence, features, X, P, Ex, Ep, Exp = preprocess.process_frame(audio_window)
                features = np.expand_dims(features, (0, 1)).astype(np.float32)
                if not silence:
                    input_data = torch.from_numpy(features).to(device)
                    QuantStub(input_data,data_config['fp32_min'],data_config['fp32_max'],symm,bits,isHW=False)
                    denoise_gru_state, denoise_output, noise_gru_state, vad_gru_state, vad_out = model(input_data,vad_gru_state,noise_gru_state,denoise_gru_state)
                    vad_gru_state = torch.squeeze(vad_gru_state, axis=1)
                    noise_gru_state = torch.squeeze(noise_gru_state, axis=1)
                    denoise_gru_state = torch.squeeze(denoise_gru_state, axis=1)

                    denoise_output = torch.squeeze(denoise_output)
                    denoise_output_np = denoise_output.to("cpu").detach().numpy()
                    denoised_audio_tmp = preprocess.post_process(silence, denoise_output_np, X, P, Ex, Ep, Exp)
                    denoised_audio_tmp = np.rint(denoised_audio_tmp)
                    final_denoised_audio.append(denoised_audio_tmp)
                else:
                    final_denoised_audio.append(np.zeros([preprocess.FRAME_SIZE], dtype=np.int16))
            denoised_audio = np.concatenate(final_denoised_audio, axis=0)
            resample_clean = librosa.resample(clean_speech, orig_sr=48000, target_sr=16000)
            resample_denoised = librosa.resample(denoised_audio, orig_sr=48000, target_sr=16000)
            sum_pesq += pesq.pesq(ref=resample_clean, deg=resample_denoised, fs=16000, mode='wb')
            num_file=num_file+1
            if calibration:
                print(str(num_file_cal)+" cal files avg pesq = "+str(sum_pesq/num_file_cal))
            else:
                print(str(num_file)+" files avg pesq = "+str(sum_pesq/num_file))
        avg_pesq = sum_pesq / num_file
    print(f"Result FQ acc is %f" % avg_pesq)
    return avg_pesq

"""
Define Hardware(HW) Quantization model inference
"""
def inference_HW(model, dataloaders, data_config, device, symm=True, bits=8, calibration=False):
    model.eval()
    model.to(device)
    print("Start accuracy estimator inference")
    dataloader=copy.deepcopy(dataloaders)
    print("state initialize")
    vad_gru_state = torch.zeros((1, 24)).to(device)
    noise_gru_state = torch.zeros((1, 48)).to(device)
    denoise_gru_state = torch.zeros((1, 96)).to(device)
    sum_pesq = 0.0
    final_denoised_audio = []
    num_file=0
    num_file_cal=0

    with torch.no_grad():
        for clean_speech, noisy_speech in tqdm(dataloader):
            if calibration:
                if num_file % 8 == 0:
                    num_file_cal = num_file_cal + 1
                    #every 8 file go forward function, otherwise pass
                    pass
                else:
                    num_file = num_file +1
                    continue
            final_denoised_audio=[]
            preprocess = RNNoisePreProcess(training=False)
            num_samples = len(noisy_speech) // preprocess.FRAME_SIZE
            for i in range(num_samples):
                audio_window = noisy_speech[i * RNNoisePreProcess.FRAME_SIZE:
                                        i * RNNoisePreProcess.FRAME_SIZE + RNNoisePreProcess.FRAME_SIZE]
                silence, features, X, P, Ex, Ep, Exp = preprocess.process_frame(audio_window)
                features = np.expand_dims(features, (0, 1)).astype(np.float32)
                if not silence:
                    input_data = torch.from_numpy(features).to(device)
                    #QuantStub(input_data,data_config['fp32_min'],data_config['fp32_max'],symm,bits,isHW=False)
                    denoise_gru_state, denoise_output, noise_gru_state, vad_gru_state, vad_out = model(input_data,vad_gru_state,noise_gru_state,denoise_gru_state)
                    vad_gru_state = torch.squeeze(vad_gru_state, axis=1)
                    noise_gru_state = torch.squeeze(noise_gru_state, axis=1)
                    denoise_gru_state = torch.squeeze(denoise_gru_state, axis=1)

                    denoise_output = torch.squeeze(denoise_output)
                    denoise_output_np = denoise_output.to("cpu").detach().numpy()
                    denoised_audio_tmp = preprocess.post_process(silence, denoise_output_np, X, P, Ex, Ep, Exp)
                    denoised_audio_tmp = np.rint(denoised_audio_tmp)
                    final_denoised_audio.append(denoised_audio_tmp)
                else:
                    final_denoised_audio.append(np.zeros([preprocess.FRAME_SIZE], dtype=np.int16))
            denoised_audio = np.concatenate(final_denoised_audio, axis=0)
            resample_clean = librosa.resample(clean_speech, orig_sr=48000, target_sr=16000)
            resample_denoised = librosa.resample(denoised_audio, orig_sr=48000, target_sr=16000)
            sum_pesq += pesq.pesq(ref=resample_clean, deg=resample_denoised, fs=16000, mode='wb')
            num_file=num_file+1
            if calibration:
                print(str(num_file_cal)+" cal files avg pesq = "+str(sum_pesq/num_file_cal))
            else:
                print(str(num_file)+" files avg pesq = "+str(sum_pesq/num_file))
        avg_pesq = sum_pesq / num_file
    print(f"Result accuracy estimator acc is %f" % avg_pesq)
    return avg_pesq

def inference_Backend(interpreter, dataloader, data_config, device, symm=True, bits=8):
    print("Start inference Backend")
    device="cpu"
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    """
    main input index assign
    """
    feature_input_idx = 0
    input_scale = input_details[feature_input_idx]['quantization_parameters']['scales'][0]
    input_zp = input_details[feature_input_idx]['quantization_parameters']['zero_points'][0]

    """
    state input index assign
    """
    vad_gru_state_idx = 1
    vad_gru_scale = input_details[vad_gru_state_idx]['quantization_parameters']['scales'][0]
    vad_gru_zp = input_details[vad_gru_state_idx]['quantization_parameters']['zero_points'][0]
    vad_gru_shape = tuple(input_details[vad_gru_state_idx]['shape'])
    input_dtype = input_details[vad_gru_state_idx]['dtype']
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    vad_gru_state = torch.zeros(vad_gru_shape).div_(vad_gru_scale).round_().add_(vad_gru_zp).clamp_(input_qmin, input_qmax)
    vad_gru_state = input_dtype(vad_gru_state.to(device).numpy())

    noise_gru_state_idx = 2
    noise_gru_scale = input_details[noise_gru_state_idx]['quantization_parameters']['scales'][0]
    noise_gru_zp = input_details[noise_gru_state_idx]['quantization_parameters']['zero_points'][0]
    noise_gru_shape = tuple(input_details[noise_gru_state_idx]['shape'])
    input_dtype = input_details[noise_gru_state_idx]['dtype']
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    noise_gru_state = torch.zeros(noise_gru_shape).div_(noise_gru_scale).round_().add_(noise_gru_zp).clamp_(input_qmin, input_qmax)
    noise_gru_state = input_dtype(noise_gru_state.to(device).numpy())

    denoise_gru_state_idx = 3
    denoise_gru_scale = input_details[denoise_gru_state_idx]['quantization_parameters']['scales'][0]
    denoise_gru_zp = input_details[denoise_gru_state_idx]['quantization_parameters']['zero_points'][0]
    denoise_gru_shape = tuple(input_details[denoise_gru_state_idx]['shape'])
    input_dtype = input_details[denoise_gru_state_idx]['dtype']
    input_qmin = np.iinfo(input_dtype).min
    input_qmax = np.iinfo(input_dtype).max
    denoise_gru_state = torch.zeros(denoise_gru_shape).div_(denoise_gru_scale).round_().add_(denoise_gru_zp).clamp_(input_qmin, input_qmax)
    denoise_gru_state = input_dtype(denoise_gru_state.to(device).numpy())

    """
    main output index assign
    """
    output_idx=1
    """
    state output index assign
    """
    vad_gru_output_s24=3
    noise_gru_output_s48=2
    denoise_gru_output_s96=0

    """
    metrics var
    """
    sum_pesq=0.0
    num_file=0

    for clean_speech, noisy_speech in tqdm(dataloader):
        preprocess = RNNoisePreProcess(training=False)
        num_samples = len(noisy_speech) // preprocess.FRAME_SIZE
        final_denoised_audio=[]
        for i in range(num_samples):
            audio_window = noisy_speech[i * RNNoisePreProcess.FRAME_SIZE:
                                        i * RNNoisePreProcess.FRAME_SIZE + RNNoisePreProcess.FRAME_SIZE]
            silence, features, X, P, Ex, Ep, Exp = preprocess.process_frame(audio_window)
            features = np.expand_dims(features, (0, 1)).astype(np.float32)
            input_dtype = input_details[feature_input_idx]['dtype']
            input_qmin = np.iinfo(input_dtype).min
            input_qmax = np.iinfo(input_dtype).max
            if not silence:
                input_data = torch.from_numpy(features).to(device)
                input_data.div_(input_scale).round_().add_(input_zp).clamp_(input_qmin, input_qmax)
                input_data = input_dtype(input_data.numpy())
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.set_tensor(input_details[1]['index'], vad_gru_state)
                interpreter.set_tensor(input_details[2]['index'], noise_gru_state)
                interpreter.set_tensor(input_details[3]['index'], denoise_gru_state)
                interpreter.invoke()
                """
                dealing with main output
                """
                result = interpreter.get_tensor(output_details[output_idx]['index'])
                output_scale = output_details[output_idx]['quantization_parameters']['scales'][0]
                output_zp = output_details[output_idx]['quantization_parameters']['zero_points'][0]
                result = output_scale * (result.astype(np.float32) - output_zp)
                denoise_output = torch.from_numpy(result)
                denoise_output = torch.squeeze(denoise_output)
                denoise_output_np = denoise_output.to("cpu").detach().numpy()
                denoised_audio_tmp = preprocess.post_process(silence, denoise_output_np, X, P, Ex, Ep, Exp)
                denoised_audio_tmp = np.rint(denoised_audio_tmp)
                final_denoised_audio.append(denoised_audio_tmp)
                """
                state output for recursive
                """
                result = interpreter.get_tensor(output_details[vad_gru_output_s24]['index'])
                output_scale = output_details[vad_gru_output_s24]['quantization_parameters']['scales'][0]
                output_zp = output_details[vad_gru_output_s24]['quantization_parameters']['zero_points'][0]
                #result = output_scale * (result.astype(np.float64) - output_zp)
                vad_gru_state = torch.from_numpy(result)
                vad_gru_state=torch.squeeze(vad_gru_state, axis=1)

                result = interpreter.get_tensor(output_details[noise_gru_output_s48]['index'])
                output_scale = output_details[noise_gru_output_s48]['quantization_parameters']['scales'][0]
                output_zp = output_details[noise_gru_output_s48]['quantization_parameters']['zero_points'][0]
                #result = output_scale * (result.astype(np.float64) - output_zp)
                noise_gru_state = torch.from_numpy(result)
                noise_gru_state=torch.squeeze(noise_gru_state, axis=1)
                
                result = interpreter.get_tensor(output_details[denoise_gru_output_s96]['index'])
                output_scale = output_details[denoise_gru_output_s96]['quantization_parameters']['scales'][0]
                output_zp = output_details[denoise_gru_output_s96]['quantization_parameters']['zero_points'][0]
                #result = output_scale * (result.astype(np.float64) - output_zp)
                denoise_gru_state = torch.from_numpy(result)
                denoise_gru_state = torch.squeeze(denoise_gru_state, axis=1)
            else:
                final_denoised_audio.append(np.zeros([preprocess.FRAME_SIZE], dtype=np.int16))
        denoised_audio = np.concatenate(final_denoised_audio, axis=0)
        resample_clean = librosa.resample(clean_speech, orig_sr=48000, target_sr=16000)
        resample_denoised = librosa.resample(denoised_audio, orig_sr=48000, target_sr=16000)
        sum_pesq += pesq.pesq(ref=resample_clean, deg=resample_denoised, fs=16000, mode='wb')
        num_file=num_file+1
        print(sum_pesq/num_file)

    avg_pesq = sum_pesq / num_file
    print(f"Result Backend avg_pesq is %f" % avg_pesq)
    return avg_pesq


def inference_c(interpreter, dataloader, out_path):
    print("Start inference c Backend")
    
    result_idx=1
    result_s24=0
    result_s48=2
    result_s96=3
    # result_s1=1
    result_s1=4

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization_parameters']['scales'][0]
    input_zp = input_details[0]['quantization_parameters']['zero_points'][0]
    output_scale = output_details[result_idx]['quantization_parameters']['scales'][0]
    output_zp = output_details[result_idx]['quantization_parameters']['zero_points'][0]
    output_dtype = output_details[result_idx]["dtype"]
    shape = output_details[result_idx]['shape']

    sum_pesq=0.0

    correct_count = 0
    dd=1
    data_count = 0
    for clean_speech, noisy_speech in tqdm(dataloader):
        preprocess = RNNoisePreProcess(training=False)
        num_samples = len(noisy_speech) // preprocess.FRAME_SIZE
        final_denoised_audio = []
        for i in range(num_samples):
            audio_window = noisy_speech[i * RNNoisePreProcess.FRAME_SIZE:
                                        i * RNNoisePreProcess.FRAME_SIZE + RNNoisePreProcess.FRAME_SIZE]
            silence, features, X, P, Ex, Ep, Exp = preprocess.process_frame(audio_window)
            features = np.expand_dims(features, (0, 1)).astype(np.float32)
            if not silence:
                with open (out_path + "/out_" + str(data_count) + "_" + str(i) + '.bin', 'rb') as fi:
                    res = np.fromfile(fi,output_dtype).reshape(shape)
                tflite_out = res
                result = output_scale * (tflite_out.astype(np.float32) - output_zp)
                denoise_output = torch.from_numpy(result)
                """
                result is here
                """
                denoise_output = torch.squeeze(denoise_output)
                denoise_output_np = denoise_output.to("cpu").detach().numpy()
                denoised_audio_tmp = preprocess.post_process(silence, denoise_output_np, X, P, Ex, Ep, Exp)
                denoised_audio_tmp = np.rint(denoised_audio_tmp)
                final_denoised_audio.append(denoised_audio_tmp)

            else:
                final_denoised_audio.append(np.zeros([preprocess.FRAME_SIZE], dtype=np.int16))

        denoised_audio = np.concatenate(final_denoised_audio, axis=0)
        resample_clean = librosa.resample(clean_speech, orig_sr=48000, target_sr=16000)
        resample_denoised = librosa.resample(denoised_audio, orig_sr=48000, target_sr=16000)
        sum_pesq += pesq.pesq(ref=resample_clean, deg=resample_denoised, fs=16000, mode='wb')
        print(sum_pesq/dd)
        dd=dd+1
        data_count = data_count + 1

    avg_pesq = sum_pesq / 824.0
    print(avg_pesq)
    print(f"Result Backend avg_pesq is %f" % avg_pesq)
    return avg_pesq

def forward_one(model, dataloaders, device):
    model.eval()
    model.to(device)
    dataloader=copy.deepcopy(dataloaders)
    print("Start one FP32 inference")

    print("state initialize")
    vad_gru_state = torch.zeros((1, 24)).to(device)
    noise_gru_state = torch.zeros((1, 48)).to(device)
    denoise_gru_state = torch.zeros((1, 96)).to(device)
    sum_pesq = 0.0
    final_denoised_audio = []
    count=0.0

    with torch.no_grad():
        for clean_speech, noisy_speech in dataloader:
            final_denoised_audio=[]
            preprocess = RNNoisePreProcess(training=False)
            num_samples = len(noisy_speech) // preprocess.FRAME_SIZE
            for i in range(num_samples):
                audio_window = noisy_speech[i * RNNoisePreProcess.FRAME_SIZE:
                                        i * RNNoisePreProcess.FRAME_SIZE + RNNoisePreProcess.FRAME_SIZE]
                silence, features, X, P, Ex, Ep, Exp = preprocess.process_frame(audio_window)
                features = np.expand_dims(features, (0, 1)).astype(np.float32)
                if not silence:
                    input_data = torch.from_numpy(features).to(device)
                    denoise_gru_state, denoise_output, noise_gru_state, vad_gru_state, vad_out = model(input_data,vad_gru_state,noise_gru_state,denoise_gru_state)
                    return 0.0
                    vad_gru_state = torch.squeeze(vad_gru_state, axis=1)
                    noise_gru_state = torch.squeeze(noise_gru_state, axis=1)
                    denoise_gru_state = torch.squeeze(denoise_gru_state, axis=1)

                    denoise_output = torch.squeeze(denoise_output)
                    denoise_output_np = denoise_output.to("cpu").detach().numpy()
                    denoised_audio_tmp = preprocess.post_process(silence, denoise_output_np, X, P, Ex, Ep, Exp)
                    denoised_audio_tmp = np.rint(denoised_audio_tmp)
                    final_denoised_audio.append(denoised_audio_tmp)
                else:
                    final_denoised_audio.append(np.zeros([preprocess.FRAME_SIZE], dtype=np.int16))

def forward_one_Q(model, dataloaders, data_config, device, symm=True, bits=8):
    model.eval()
    model.to(device)
    calibration=False
    print("Start one FQ inference")
    dataloader=copy.deepcopy(dataloaders)
    print("state initialize")
    vad_gru_state = torch.zeros((1, 24)).to(device)
    noise_gru_state = torch.zeros((1, 48)).to(device)
    denoise_gru_state = torch.zeros((1, 96)).to(device)
    sum_pesq = 0.0
    final_denoised_audio = []
    num_file=0
    num_file_cal=0

    with torch.no_grad():
        for clean_speech, noisy_speech in tqdm(dataloader):
            if calibration:
                if num_file % 8 == 0:
                    num_file_cal = num_file_cal + 1
                    #every 8 file go forward function, otherwise pass
                    pass
                else:
                    num_file = num_file +1
                    continue
            final_denoised_audio=[]
            preprocess = RNNoisePreProcess(training=False)
            num_samples = len(noisy_speech) // preprocess.FRAME_SIZE
            for i in range(num_samples):
                audio_window = noisy_speech[i * RNNoisePreProcess.FRAME_SIZE:
                                        i * RNNoisePreProcess.FRAME_SIZE + RNNoisePreProcess.FRAME_SIZE]
                silence, features, X, P, Ex, Ep, Exp = preprocess.process_frame(audio_window)
                features = np.expand_dims(features, (0, 1)).astype(np.float32)
                if not silence:
                    input_data = torch.from_numpy(features).to(device)
                    QuantStub(input_data,data_config['fp32_min'],data_config['fp32_max'],symm,bits,isHW=False)
                    denoise_gru_state, denoise_output, noise_gru_state, vad_gru_state, vad_out = model(input_data,vad_gru_state,noise_gru_state,denoise_gru_state)
                    return 0.0
                    vad_gru_state = torch.squeeze(vad_gru_state, axis=1)
                    noise_gru_state = torch.squeeze(noise_gru_state, axis=1)
                    denoise_gru_state = torch.squeeze(denoise_gru_state, axis=1)

                    denoise_output = torch.squeeze(denoise_output)
                    denoise_output_np = denoise_output.to("cpu").detach().numpy()
                    denoised_audio_tmp = preprocess.post_process(silence, denoise_output_np, X, P, Ex, Ep, Exp)
                    denoised_audio_tmp = np.rint(denoised_audio_tmp)
                    final_denoised_audio.append(denoised_audio_tmp)
                else:
                    final_denoised_audio.append(np.zeros([preprocess.FRAME_SIZE], dtype=np.int16))
