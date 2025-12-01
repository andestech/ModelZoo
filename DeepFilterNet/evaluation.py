import os
import torch
import numpy as np
from tqdm import tqdm
import librosa
import pesq
import copy
import math
from .df.enhance import df_features
from .df.modules import DfOp, Mask
from .df.utils import as_complex
from torchaudio.transforms import Resample
from .df.io import get_resample_params
from .df.evaluation_utils import pesq_
from .dataset import dataset_cfg
from libdf import DF

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
    df_state = DF(
        sr=48000,
        fft_size=960,
        hop_size=480,
        nb_bands=32,
        min_nb_erb_freqs=2,
    )
    df_op = torch.jit.script(
        DfOp(
            96,
            5,
            1,
            freq_bins=481,
            method="real_unfold",
        )
    )
    now_dir=os.path.dirname(__file__)
    erb_inv_fb = torch.load(now_dir + '/Model/mask.pt').to(device)
    mask = Mask(erb_inv_fb, post_filter=False).to(device)
    RESAMPLE_METHOD = "sinc_fast"
    params = get_resample_params(RESAMPLE_METHOD)
    resampler = Resample(48000, 16000, **params)
    print("state initialize")
    nb_df = 96
    sum_pesq = 0.0
    final_denoised_audio = []
    count=0.0
    data_length = 96000
    with torch.no_grad():
        for clean_speech, noisy_speech in dataloader:
            final_denoised_audio=[]
            num_samples = math.ceil(noisy_speech.shape[1] / data_length)
            encoder_gru_state = torch.zeros(24, 1, 64).to(device)
            dfdecoder_gru_state = torch.zeros(16, 1, 64).to(device)
            for i in range(num_samples):
                noisy_data = noisy_speech[:, data_length * i : data_length * (i+1)]
                if noisy_data.shape[1] <= data_length:
                    zeros = torch.zeros(1, data_length - noisy_data.shape[1])
                    noisy_data = torch.cat((noisy_data, zeros), dim=1)
                    clean_speech = torch.cat((clean_speech, zeros), dim=1)
                spec, erb_feat, spec_feat = df_features(noisy_data, df_state, nb_df, device=device)
                m, lsnr, df_coefs, df_alpha, encoder_gru_state, dfdecoder_gru_state = model(erb_feat, spec_feat, encoder_gru_state, dfdecoder_gru_state)
                spec = mask(spec, m)
                spec = df_op(spec, df_coefs, df_alpha)
                spec = spec.squeeze(0)
                final_denoised_audio.append(spec)
            denoised_audio = torch.cat(final_denoised_audio, axis=1)
            denoised_audio = df_state.synthesis(as_complex(denoised_audio).cpu().numpy())[0]
            clean_speech = df_state.synthesis(df_state.analysis(clean_speech.numpy()))[0]
            resample_denoised = resampler.forward(torch.as_tensor(denoised_audio).clone())
            resample_clean = resampler.forward(torch.as_tensor(clean_speech).clone())
            sum_pesq += pesq_(resample_clean, resample_denoised, 16000)
            count = count + 1
        avg_pesq=sum_pesq/count
        print(f"Result fp32 acc is %f" % avg_pesq)
    return avg_pesq

"""
define Fake Quantization (FQ) model inferences
"""
def inference_FQ(model, dataloaders, data_config, device, symm=True, bits=8, calibration=False):
    model.eval()
    model.to(device)
    dataloader=copy.deepcopy(dataloaders)
    print("Start FQ inference")
    input_yaml=dataset_cfg()
    test_wav_num=input_yaml['test_wav_num']
    df_state = DF(
        sr=48000,
        fft_size=960,
        hop_size=480,
        nb_bands=32,
        min_nb_erb_freqs=2,
    )
    df_op = torch.jit.script(
        DfOp(
            96,
            5,
            1,
            freq_bins=481,
            method="real_unfold",
        )
    )
    now_dir=os.path.dirname(__file__)
    erb_inv_fb = torch.load(now_dir + '/Model/mask.pt').to(device)
    mask = Mask(erb_inv_fb, post_filter=False).to(device)
    RESAMPLE_METHOD = "sinc_fast"
    params = get_resample_params(RESAMPLE_METHOD)
    resampler = Resample(48000, 16000, **params)
    print("state initialize")
    nb_df = 96
    sum_pesq = 0.0
    final_denoised_audio = []
    count=0.0
    data_length = 96000
    with torch.no_grad():
        for clean_speech, noisy_speech in tqdm(dataloader):
            final_denoised_audio=[]
            num_samples = math.ceil(noisy_speech.shape[1] / data_length)
            encoder_gru_state = torch.zeros(24, 1, 64).to(device)
            dfdecoder_gru_state = torch.zeros(16, 1, 64).to(device)
            for i in range(num_samples):
                noisy_data = noisy_speech[:, data_length * i : data_length * (i+1)]
                if noisy_data.shape[1] <= data_length:
                    zeros = torch.zeros(1, data_length - noisy_data.shape[1])
                    noisy_data = torch.cat((noisy_data, zeros), dim=1)
                    clean_speech = torch.cat((clean_speech, zeros), dim=1)
                spec, erb_feat, spec_feat = df_features(noisy_data, df_state, nb_df, device=device)
                m, lsnr, df_coefs, df_alpha, encoder_gru_state, dfdecoder_gru_state = model(erb_feat, spec_feat, encoder_gru_state, dfdecoder_gru_state)
                spec = mask(spec, m)
                spec = df_op(spec, df_coefs, df_alpha)
                spec = spec.squeeze(0)
                final_denoised_audio.append(spec)
            denoised_audio = torch.cat(final_denoised_audio, axis=1)
            denoised_audio = df_state.synthesis(as_complex(denoised_audio).cpu().numpy())[0]
            clean_speech = df_state.synthesis(df_state.analysis(clean_speech.numpy()))[0]
            resample_denoised = resampler.forward(torch.as_tensor(denoised_audio).clone())
            resample_clean = resampler.forward(torch.as_tensor(clean_speech).clone())
            sum_pesq += pesq_(resample_clean, resample_denoised, 16000)
            count = count + 1
            print(sum_pesq/count)
            if calibration and count==20:
                print("calibration collect")
                return 0.0
            elif not calibration and count==test_wav_num:
                avg_pesq=sum_pesq/count
                print(f"Result partial FQ acc is %f" % avg_pesq)
                return avg_pesq
        avg_pesq=sum_pesq/count
        print(f"Result FQ acc is %f" % avg_pesq)
    return avg_pesq

"""
Define Hardware(HW) Quantization model inference
"""

def inference_HW(model, dataloaders, data_config, device, symm=True, bits=8, calibration=False):
    model.eval()
    model.to(device)
    dataloader=copy.deepcopy(dataloaders)
    print("Start accuracy estimator inference")
    input_yaml=dataset_cfg()
    test_wav_num=input_yaml['test_wav_num']
    df_state = DF(
        sr=48000,
        fft_size=960,
        hop_size=480,
        nb_bands=32,
        min_nb_erb_freqs=2,
    )
    df_op = torch.jit.script(
        DfOp(
            96,
            5,
            1,
            freq_bins=481,
            method="real_unfold",
        )
    )
    now_dir=os.path.dirname(__file__)
    erb_inv_fb = torch.load(now_dir + '/Model/mask.pt').to(device)
    mask = Mask(erb_inv_fb, post_filter=False).to(device)
    RESAMPLE_METHOD = "sinc_fast"
    params = get_resample_params(RESAMPLE_METHOD)
    resampler = Resample(48000, 16000, **params)
    print("state initialize")
    nb_df = 96
    sum_pesq = 0.0
    final_denoised_audio = []
    count=0.0
    data_length = 96000
    with torch.no_grad():
        for clean_speech, noisy_speech in tqdm(dataloader):
            final_denoised_audio=[]
            num_samples = math.ceil(noisy_speech.shape[1] / data_length)
            encoder_gru_state = torch.zeros(24, 1, 64).to(device)
            dfdecoder_gru_state = torch.zeros(16, 1, 64).to(device)
            for i in range(num_samples):
                noisy_data = noisy_speech[:, data_length * i : data_length * (i+1)]
                if noisy_data.shape[1] <= data_length:
                    zeros = torch.zeros(1, data_length - noisy_data.shape[1])
                    noisy_data = torch.cat((noisy_data, zeros), dim=1)
                    clean_speech = torch.cat((clean_speech, zeros), dim=1)
                spec, erb_feat, spec_feat = df_features(noisy_data, df_state, nb_df, device=device)
                m, lsnr, df_coefs, df_alpha, encoder_gru_state, dfdecoder_gru_state = model(erb_feat, spec_feat, encoder_gru_state, dfdecoder_gru_state)
                spec = mask(spec, m)
                spec = df_op(spec, df_coefs, df_alpha)
                spec = spec.squeeze(0)
                final_denoised_audio.append(spec)
            denoised_audio = torch.cat(final_denoised_audio, axis=1)
            denoised_audio = df_state.synthesis(as_complex(denoised_audio).cpu().numpy())[0]
            clean_speech = df_state.synthesis(df_state.analysis(clean_speech.numpy()))[0]
            resample_denoised = resampler.forward(torch.as_tensor(denoised_audio).clone())
            resample_clean = resampler.forward(torch.as_tensor(clean_speech).clone())
            sum_pesq += pesq_(resample_clean, resample_denoised, 16000)
            count = count + 1
            print(sum_pesq/count)
            if calibration and count==20:
                print("calibration collect")
                return 0.0
            elif not calibration and count==test_wav_num:
                avg_pesq=sum_pesq/count
                print(f"Result partial accuracy estimator acc is %f" % avg_pesq)
                return avg_pesq
        avg_pesq=sum_pesq/count
        print(f"Result accuracy estimator acc is %f" % avg_pesq)
    return avg_pesq

def inference_Backend(interpreter, dataloaders, data_config, device, symm=True, bits=8):
    dataloader=copy.deepcopy(dataloaders)
    print("Start Backend inference")
    # input_idx
    feature_input_1_idx = 0
    feature_input_2_idx = 1
    encoder_gru_input_idx = 2
    dfdecoder_gru_input_idx = 3
    # output_idx
    m_output_idx = 0
    lsnr_output_idx = 1
    df_coefs_output_idx = 2
    df_alpha_output_idx = 3
    encoder_gru_output_idx = 4
    dfdecoder_gru_output_idx = 5
    """
    main input index assign
    """
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_1_scale = input_details[feature_input_1_idx]['quantization_parameters']['scales'][0]
    input_1_zp = input_details[feature_input_1_idx]['quantization_parameters']['zero_points'][0]
    input_1_dtype = input_details[feature_input_1_idx]['dtype']
    input_1_qmin = np.iinfo(input_1_dtype).min
    input_1_qmax = np.iinfo(input_1_dtype).max
    input_2_scale = input_details[feature_input_2_idx]['quantization_parameters']['scales'][0]
    input_2_zp = input_details[feature_input_2_idx]['quantization_parameters']['zero_points'][0]
    input_2_dtype = input_details[feature_input_2_idx]['dtype']
    input_2_qmin = np.iinfo(input_2_dtype).min
    input_2_qmax = np.iinfo(input_2_dtype).max

    input_yaml=dataset_cfg()
    test_wav_num=input_yaml['test_wav_num']
    df_state = DF(
        sr=48000,
        fft_size=960,
        hop_size=480,
        nb_bands=32,
        min_nb_erb_freqs=2,
    )
    df_op = torch.jit.script(
        DfOp(
            96,
            5,
            1,
            freq_bins=481,
            method="real_unfold",
        )
    )
    now_dir=os.path.dirname(__file__)
    erb_inv_fb = torch.load(now_dir + '/Model/mask.pt').to(device)
    mask = Mask(erb_inv_fb, post_filter=False).to(device)
    RESAMPLE_METHOD = "sinc_fast"
    params = get_resample_params(RESAMPLE_METHOD)
    resampler = Resample(48000, 16000, **params)
    print("state initialize")
    nb_df = 96
    sum_pesq = 0.0
    final_denoised_audio = []
    count=0.0
    data_length = 96000
    with torch.no_grad():
        for clean_speech, noisy_speech in tqdm(dataloader):
            final_denoised_audio=[]
            num_samples = math.ceil(noisy_speech.shape[1] / data_length)
            """
            state input index assign
            """
            encoder_gru_scale = input_details[encoder_gru_input_idx]['quantization_parameters']['scales'][0]
            encoder_gru_zp = input_details[encoder_gru_input_idx]['quantization_parameters']['zero_points'][0]
            input_3_dtype = input_details[encoder_gru_input_idx]['dtype']
            input_3_qmin = np.iinfo(input_3_dtype).min
            input_3_qmax = np.iinfo(input_3_dtype).max
            encoder_gru_shape = tuple(input_details[encoder_gru_input_idx]['shape'])
            encoder_gru_state = torch.zeros(encoder_gru_shape).div_(encoder_gru_scale).round_().add_(encoder_gru_zp).clamp_(input_3_qmin, input_3_qmax)
            encoder_gru_state = input_3_dtype(encoder_gru_state.to(device).numpy())
            dfdecoder_gru_scale = input_details[dfdecoder_gru_input_idx]['quantization_parameters']['scales'][0]
            dfdecoder_gru_zp = input_details[dfdecoder_gru_input_idx]['quantization_parameters']['zero_points'][0]
            input_4_dtype = input_details[dfdecoder_gru_input_idx]['dtype']
            input_4_qmin = np.iinfo(input_4_dtype).min
            input_4_qmax = np.iinfo(input_4_dtype).max
            dfdecoder_gru_shape = tuple(input_details[dfdecoder_gru_input_idx]['shape'])
            dfdecoder_gru_state = torch.zeros(dfdecoder_gru_shape).div_(dfdecoder_gru_scale).round_().add_(dfdecoder_gru_zp).clamp_(input_4_qmin, input_4_qmax)
            dfdecoder_gru_state = input_4_dtype(dfdecoder_gru_state.to(device).numpy())

            for i in range(num_samples):
                noisy_data = noisy_speech[:, data_length * i : data_length * (i+1)]
                if noisy_data.shape[1] <= data_length:
                    zeros = torch.zeros(1, data_length - noisy_data.shape[1])
                    noisy_data = torch.cat((noisy_data, zeros), dim=1)
                    clean_speech = torch.cat((clean_speech, zeros), dim=1)
                spec, erb_feat, spec_feat = df_features(noisy_data, df_state, nb_df, device=device)

                erb_feat.div_(input_1_scale).round_().add_(input_1_zp).clamp_(input_1_qmin, input_1_qmax)
                erb_feat = input_1_dtype(erb_feat.numpy())
                spec_feat.div_(input_2_scale).round_().add_(input_2_zp).clamp_(input_2_qmin, input_2_qmax)
                spec_feat = input_2_dtype(spec_feat.numpy())
                # if i > 0:
                #     encoder_gru_state.div_(encoder_gru_scale).round_().add_(encoder_gru_zp).clamp_(input_3_qmin, input_3_qmax)
                #     encoder_gru_state = input_3_dtype(encoder_gru_state.numpy())
                #     dfdecoder_gru_state.div_(dfdecoder_gru_scale).round_().add_(dfdecoder_gru_zp).clamp_(input_4_qmin, input_4_qmax)
                #     dfdecoder_gru_state = input_4_dtype(dfdecoder_gru_state.numpy())
                interpreter.set_tensor(input_details[0]['index'], erb_feat.transpose(0, 2, 3, 1))
                interpreter.set_tensor(input_details[1]['index'], spec_feat.transpose(0, 4, 2, 3, 1))
                interpreter.set_tensor(input_details[2]['index'], encoder_gru_state)
                interpreter.set_tensor(input_details[3]['index'], dfdecoder_gru_state)
                interpreter.invoke()

                m = interpreter.get_tensor(output_details[m_output_idx]['index'])
                output_scale = output_details[m_output_idx]['quantization_parameters']['scales'][0]
                output_zp = output_details[m_output_idx]['quantization_parameters']['zero_points'][0]
                m = output_scale * (m.astype(np.float32) - output_zp)
                m = torch.from_numpy(m.transpose(0, 3, 1, 2))

                df_coefs = interpreter.get_tensor(output_details[df_coefs_output_idx]['index'])
                output_scale = output_details[df_coefs_output_idx]['quantization_parameters']['scales'][0]
                output_zp = output_details[df_coefs_output_idx]['quantization_parameters']['zero_points'][0]
                df_coefs = output_scale * (df_coefs.astype(np.float32) - output_zp)
                df_coefs = torch.from_numpy(df_coefs.transpose(0, 1, 2, 4, 3))

                df_alpha = interpreter.get_tensor(output_details[df_alpha_output_idx]['index'])
                output_scale = output_details[df_alpha_output_idx]['quantization_parameters']['scales'][0]
                output_zp = output_details[df_alpha_output_idx]['quantization_parameters']['zero_points'][0]
                df_alpha = output_scale * (df_alpha.astype(np.float32) - output_zp)
                df_alpha = torch.from_numpy(df_alpha)
                
                encoder_gru_state = interpreter.get_tensor(output_details[encoder_gru_output_idx]['index'])
                # output_scale = output_details[encoder_gru_output_idx]['quantization_parameters']['scales'][0]
                # output_zp = output_details[encoder_gru_output_idx]['quantization_parameters']['zero_points'][0]
                # encoder_gru_state = output_scale * (encoder_gru_state.astype(np.float32) - output_zp)
                # encoder_gru_state = torch.from_numpy(encoder_gru_state)

                dfdecoder_gru_state = interpreter.get_tensor(output_details[dfdecoder_gru_output_idx]['index'])
                # output_scale = output_details[dfdecoder_gru_output_idx]['quantization_parameters']['scales'][0]
                # output_zp = output_details[dfdecoder_gru_output_idx]['quantization_parameters']['zero_points'][0]
                # dfdecoder_gru_state = output_scale * (dfdecoder_gru_state.astype(np.float32) - output_zp)
                # dfdecoder_gru_state = torch.from_numpy(dfdecoder_gru_state)

                spec = mask(spec, m)
                spec = df_op(spec, df_coefs, df_alpha)
                spec = spec.squeeze(0)
                final_denoised_audio.append(spec)
            denoised_audio = torch.cat(final_denoised_audio, axis=1)
            denoised_audio = df_state.synthesis(as_complex(denoised_audio).cpu().numpy())[0]
            clean_speech = df_state.synthesis(df_state.analysis(clean_speech.numpy()))[0]
            resample_denoised = resampler.forward(torch.as_tensor(denoised_audio).clone())
            resample_clean = resampler.forward(torch.as_tensor(clean_speech).clone())
            sum_pesq += pesq_(resample_clean, resample_denoised, 16000)
            count = count + 1
            print(sum_pesq/count)
            if count == test_wav_num:
                break
        avg_pesq=sum_pesq/count
        print(f"Result Backend acc is %f" % avg_pesq)
    return avg_pesq


def inference_c(interpreter, dataloader, out_path):
    # dataloader=copy.deepcopy(dataloaders)
    print("Start Backend inference")
    device = "cpu"
    # output_idx
    m_output_idx = 0
    lsnr_output_idx = 1
    df_coefs_output_idx = 2
    df_alpha_output_idx = 3
    encoder_gru_output_idx = 4
    dfdecoder_gru_output_idx = 5
    """
    main input index assign
    """
    # interpreter.allocate_tensors() # model includes custom op cannot allocate tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_yaml=dataset_cfg()
    test_wav_num=input_yaml['test_wav_num']
    df_state = DF(
        sr=48000,
        fft_size=960,
        hop_size=480,
        nb_bands=32,
        min_nb_erb_freqs=2,
    )
    df_op = torch.jit.script(
        DfOp(
            96,
            5,
            1,
            freq_bins=481,
            method="real_unfold",
        )
    )
    now_dir=os.path.dirname(__file__)
    erb_inv_fb = torch.load(now_dir + '/Model/mask.pt').to(device)
    mask = Mask(erb_inv_fb, post_filter=False).to(device)
    RESAMPLE_METHOD = "sinc_fast"
    params = get_resample_params(RESAMPLE_METHOD)
    resampler = Resample(48000, 16000, **params)
    print("state initialize")
    nb_df = 96
    sum_pesq = 0.0
    final_denoised_audio = []
    count=0
    data_length = 96000
    with torch.no_grad():
        for clean_speech, noisy_speech in tqdm(dataloader):
            final_denoised_audio=[]
            num_samples = math.ceil(noisy_speech.shape[1] / data_length)

            for i in range(num_samples):
                noisy_data = noisy_speech[:, data_length * i : data_length * (i+1)]
                if noisy_data.shape[1] <= data_length:
                    zeros = torch.zeros(1, data_length - noisy_data.shape[1])
                    noisy_data = torch.cat((noisy_data, zeros), dim=1)
                    clean_speech = torch.cat((clean_speech, zeros), dim=1)
                spec, erb_feat, spec_feat = df_features(noisy_data, df_state, nb_df, device=device)

                shape = output_details[m_output_idx]['shape']
                output_dtype = output_details[m_output_idx]['dtype']
                with open (out_path + "/out_" + str(count) + "_" + str(i) + '_0.bin', 'rb') as fi:
                    res = np.fromfile(fi, output_dtype).reshape(shape)
                m = res
                output_scale = output_details[m_output_idx]['quantization_parameters']['scales'][0]
                output_zp = output_details[m_output_idx]['quantization_parameters']['zero_points'][0]
                m = output_scale * (m.astype(np.float32) - output_zp)
                m = torch.from_numpy(m.transpose(0, 3, 1, 2))

                # shape = output_details[lsnr_output_idx]['shape']
                # output_dtype = output_details[lsnr_output_idx]['dtype']
                # with open (out_path + "/out_" + str(count) + "_" + str(i) + '_1.bin', 'rb') as fi:
                #     res = np.fromfile(fi, output_dtype).reshape(shape)
                # lsnr = res
                # output_scale = output_details[lsnr_output_idx]['quantization_parameters']['scales'][0]
                # output_zp = output_details[lsnr_output_idx]['quantization_parameters']['zero_points'][0]
                # lsnr = output_scale * (lsnr.astype(np.float32) - output_zp)
                # lsnr = torch.from_numpy(lsnr)

                shape = output_details[df_coefs_output_idx]['shape']
                output_dtype = output_details[df_coefs_output_idx]['dtype']
                with open (out_path + "/out_" + str(count) + "_" + str(i) + '_2.bin', 'rb') as fi:
                    res = np.fromfile(fi, output_dtype).reshape(shape)
                df_coefs = res
                output_scale = output_details[df_coefs_output_idx]['quantization_parameters']['scales'][0]
                output_zp = output_details[df_coefs_output_idx]['quantization_parameters']['zero_points'][0]
                df_coefs = output_scale * (df_coefs.astype(np.float32) - output_zp)
                df_coefs = torch.from_numpy(df_coefs.transpose(0, 1, 2, 4, 3))

                shape = output_details[df_alpha_output_idx]['shape']
                output_dtype = output_details[df_alpha_output_idx]['dtype']
                with open (out_path + "/out_" + str(count) + "_" + str(i) + '_3.bin', 'rb') as fi:
                    res = np.fromfile(fi, output_dtype).reshape(shape)
                df_alpha = res
                output_scale = output_details[df_alpha_output_idx]['quantization_parameters']['scales'][0]
                output_zp = output_details[df_alpha_output_idx]['quantization_parameters']['zero_points'][0]
                df_alpha = output_scale * (df_alpha.astype(np.float32) - output_zp)
                df_alpha = torch.from_numpy(df_alpha)

                # shape = output_details[encoder_gru_output_idx]['shape']
                # output_dtype = output_details[encoder_gru_output_idx]['dtype']
                # with open (out_path + "/out_" + str(count) + "_" + str(i) + '_4.bin', 'rb') as fi:
                #     res = np.fromfile(fi, output_dtype).reshape(shape)
                # encoder_gru_state = res
                # output_scale = output_details[encoder_gru_output_idx]['quantization_parameters']['scales'][0]
                # output_zp = output_details[encoder_gru_output_idx]['quantization_parameters']['zero_points'][0]
                # encoder_gru_state = output_scale * (encoder_gru_state.astype(np.float32) - output_zp)
                # encoder_gru_state = torch.from_numpy(encoder_gru_state)

                # shape = output_details[dfdecoder_gru_output_idx]['shape']
                # output_dtype = output_details[dfdecoder_gru_output_idx]['dtype']
                # with open (out_path + "/out_" + str(count) + "_" + str(i) + '_5.bin', 'rb') as fi:
                #     res = np.fromfile(fi, output_dtype).reshape(shape)
                # dfdecoder_gru_state = res
                # output_scale = output_details[dfdecoder_gru_output_idx]['quantization_parameters']['scales'][0]
                # output_zp = output_details[dfdecoder_gru_output_idx]['quantization_parameters']['zero_points'][0]
                # dfdecoder_gru_state = output_scale * (dfdecoder_gru_state.astype(np.float32) - output_zp)
                # dfdecoder_gru_state = torch.from_numpy(dfdecoder_gru_state)

                spec = mask(spec, m)
                spec = df_op(spec, df_coefs, df_alpha)
                spec = spec.squeeze(0)
                final_denoised_audio.append(spec)
            denoised_audio = torch.cat(final_denoised_audio, axis=1)
            denoised_audio = df_state.synthesis(as_complex(denoised_audio).cpu().numpy())[0]
            clean_speech = df_state.synthesis(df_state.analysis(clean_speech.numpy()))[0]
            resample_denoised = resampler.forward(torch.as_tensor(denoised_audio).clone())
            resample_clean = resampler.forward(torch.as_tensor(clean_speech).clone())
            sum_pesq += pesq_(resample_clean, resample_denoised, 16000)
            count = count + 1
            print(sum_pesq/count)
            if count == test_wav_num:
                break
        avg_pesq=sum_pesq/count
        print(f"Result Backend acc is %f" % avg_pesq)
    return avg_pesq

def forward_one(model, dataloaders, device):
    assert NotImplementedError

def forward_one_Q(model, dataloaders, data_config, device, symm=True, bits=8):
    assert NotImplementedError
