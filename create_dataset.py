import glob
import os
import argparse
import json
import librosa
import numpy as np
import csv
import torch, torchaudio
import requests
import copy
from scipy.interpolate import interp1d
import utils
import pyworld as pw
import math

#Global
NUM_MAX_SID = 255
#JVS+Zun+Sora+Me+Tsum+Tsuk+NICT*2 = 107
NUM_TRAINED_SPEAKER = 1
ZUNDAMON_SID = 101
MY_SID = 0
VAL_PER = 20

#F0 param
FRAME_LENGTH = 512
WIN_LENGTH = 256
HOP_LENGTH = 128
UNITS_BASE_PATH = "dataset_etc"

#pyworld Harvest を使ってf0を推定する
def get_f0(wav_path, frame_length=FRAME_LENGTH, win_length=WIN_LENGTH, hop_length=HOP_LENGTH):
    y, sr = librosa.load(wav_path, 24000)
    pad_width=[int((frame_length-hop_length)/2),int((frame_length-hop_length)/2)]
    y = np.pad(y, pad_width, 'reflect')
    #Get f0
    #https://librosa.org/doc/main/generated/librosa.pyin.html
    y = y.astype(np.float64)
    f0_harvest, t = pw.harvest(y, sr, frame_period = 5.5, f0_floor=71.0, f0_ceil=1000.0)
    #specのshapeに合わせる
    f0_size = math.floor((y.shape[0] - win_length) / hop_length) - 1
    x = np.linspace(0, f0_harvest.shape[0] -1, num= f0_harvest.shape[0])
    xp = np.linspace(0, f0_harvest.shape[0] -1, num=f0_size)
    fp = np.interp(xp, x, f0_harvest)
    return fp

#f0からcf0を推定する
def convert_continuos_f0(f0):
    """Convert F0 to continuous F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)

    """
    # get uv information as binary
    uv = np.float32(f0 != 0)
    # get start and end of f0
    if (f0 == 0).all():
        return uv, f0, False
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]
    # padding start and end of f0 sequence
    cf0 = copy.deepcopy(f0)
    start_idx = np.where(cf0 == start_f0)[0][0]
    end_idx = np.where(cf0 == end_f0)[0][-1]
    cf0[:start_idx] = start_f0
    cf0[end_idx:] = end_f0
    # get non-zero frame index
    nz_frames = np.where(cf0 != 0)[0]
    # perform linear interpolation
    f = interp1d(nz_frames, cf0[nz_frames])
    cf0 = f(np.arange(0, cf0.shape[0]))

    return uv, cf0, True

def create_cf0(wav_path, d_):
    filename = os.path.basename(wav_path)
    d = os.path.basename(d_)
    save_path = "{}/cF0/{}/{}".format(UNITS_BASE_PATH, d , filename[:-4])
    load_path = "{}/F0/{}/{}".format(UNITS_BASE_PATH, d , filename[:-4] + ".npy")
    print(load_path)
    f0 = np.load(load_path)
    os.makedirs(UNITS_BASE_PATH, exist_ok=True)
    os.makedirs("{}/cF0".format(UNITS_BASE_PATH), exist_ok=True)
    os.makedirs("{}/cF0/{}".format(UNITS_BASE_PATH, d), exist_ok=True)
    if os.path.isfile(save_path+ ".npy"):
        cf0 = np.load(save_path+ ".npy")
    else:
        _, cf0, _ = convert_continuos_f0(f0)
        np.save(save_path, cf0)

    return save_path + ".npy" , np.mean(cf0)

def create_f0(wav_path, d_):
    filename = os.path.basename(wav_path)
    d = os.path.basename(d_)
    save_path = "{}/F0/{}/{}".format(UNITS_BASE_PATH, d , filename[:-4])
    os.makedirs(UNITS_BASE_PATH, exist_ok=True)
    os.makedirs("{}/F0".format(UNITS_BASE_PATH), exist_ok=True)
    os.makedirs("{}/F0/{}".format(UNITS_BASE_PATH,d), exist_ok=True)
    if os.path.isfile(save_path+ ".npy"):
        pass
    else:
        f0 = get_f0(wav_path)
        np.save(save_path, f0)
    return save_path + ".npy"

#textを音素に
def mozi2phone(hubert, wav_path, d_):
    filename = os.path.basename(wav_path)
    d = os.path.basename(d_)
    save_path = "{}/units/{}/{}".format(UNITS_BASE_PATH, d , filename[:-4])
    os.makedirs(UNITS_BASE_PATH, exist_ok=True)
    os.makedirs("{}/units".format(UNITS_BASE_PATH), exist_ok=True)
    os.makedirs("{}/units/{}".format(UNITS_BASE_PATH,d), exist_ok=True)

    if os.path.isfile(save_path+ ".npy"):
        pass
    else:
        source, sr = torchaudio.load(wav_path)
        source = torchaudio.functional.resample(source, sr, 16000)
        source = source.unsqueeze(0)
        units = hubert.units(source).numpy().squeeze(0)
        np.save(save_path, units)
    return save_path + ".npy"

#filelistの1行を作成する
def create_data_line(wav, speaker_id, d, hubert):
    units = mozi2phone(hubert, wav, d)
    f0 = create_f0(wav, d)
    cf0, cf0_mean = create_cf0(wav, d)
    #wav_path | sid | unit | f0 | cf0 | cf0_mean
    one_line = "{}|{}|{}|{}|{}|{}\n".format(
        wav,
        str(speaker_id),
        units,
        f0,
        cf0,
        str(cf0_mean)
    )
    print(one_line)
    return one_line, cf0_mean

def create_json(filename, num_speakers, sr, config_path):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data['data']['training_files'] = 'filelists/' + filename + '_textful.txt'
    data['data']['validation_files'] = 'filelists/' + filename + '_textful_val.txt'
    data['data']['training_files_notext'] = 'filelists/' + filename + '_textless.txt'
    data['data']['validation_files_notext'] = 'filelists/' + filename + '_val_textless.txt'
    data['data']['sampling_rate'] = sr
    data['data']['n_speakers'] = num_speakers

    with open("./configs/" + filename + ".json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def write_configs(lists, filename):
    #推論用の空リスト　そのうち消したい
    output_file_list_textless = list()
    with open('filelists/' + filename + '_textful.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(lists[0])
    with open('filelists/' + filename + '_textful_val.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(lists[1])
    with open('filelists/' + filename + '_textless.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_textless)
    with open('filelists/' + filename + '_Correspondence.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(lists[2])

def create_dataset(filename):
    #空白話者の先頭からスタート
    speaker_id = NUM_TRAINED_SPEAKER
    #list回りの宣言とソート
    textful_dir_list = glob.glob("dataset/**/")
    textful_dir_list.sort()
    Correspondence_list = list()
    output_file_list = list()
    output_file_list_val = list()
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")

    for d in textful_dir_list:
        d = d[:-1]
        wav_file_list = glob.glob(d+"/wav/*.wav")
        wav_file_list.sort()
        if len(wav_file_list) == 0:
            continue
        counter = 0
        cf0_mean_t = 0.0
        for wav in wav_file_list:
            one_line, cf0_mean = create_data_line(wav, speaker_id, d, hubert)
            if counter % VAL_PER != 0:
                output_file_list.append(one_line)
            else:
                output_file_list_val.append(one_line)
            counter = counter +1
            cf0_mean_t = cf0_mean_t + cf0_mean
        cf0_mean_t = cf0_mean_t / counter
        Correspondence_list.append(str(speaker_id)+"|"+ str(cf0_mean_t) + "|" + os.path.basename(d) + "\n")
        speaker_id = speaker_id + 1
        if speaker_id > NUM_MAX_SID:
            break
    
    lists = [output_file_list, output_file_list_val, Correspondence_list]
    write_configs(lists, filename)
    return NUM_MAX_SID

def create_dataset_zundamon(filename):
    #list回りの宣言とソート
    textful_dir_list = glob.glob("dataset/**/")
    textful_dir_list.sort()
    Correspondence_list = list()
    output_file_list = list()
    output_file_list_val = list()
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")
    #paths
    my_path = "dataset/00_myvoice"
    zundamon_path = "dataset/1205_zundamon"

    #set list wav and text
    #myvoice
    speaker_id = MY_SID
    d = my_path
    wav_file_list = glob.glob(d + "/wav/*.wav")
    wav_file_list.sort()
    if len(wav_file_list) == 0:
        print("Error" + d + "/wav に音声データがありません")
        exit()
    counter = 0
    cf0_mean_t = 0.0
    for wav in wav_file_list:
        one_line, cf0_mean = create_data_line(wav, speaker_id, d, hubert)
        if counter % VAL_PER != 0:
            output_file_list.append(one_line)
        else:
            output_file_list_val.append(one_line)
        counter = counter +1
        cf0_mean_t = cf0_mean_t + cf0_mean
    cf0_mean_t = cf0_mean_t / counter
    Correspondence_list.append(str(speaker_id)+"|"+ str(cf0_mean_t) + "|" + os.path.basename(d) + "\n")

    speaker_id = ZUNDAMON_SID
    d = zundamon_path
    wav_file_list = glob.glob(d + "/wav/*.wav")
    wav_file_list.sort()
    if len(wav_file_list) == 0:
        print("Error" + d + "/wav に音声データがありません")
        exit()
    counter = 0
    cf0_mean_t = 0.0
    for wav in wav_file_list:
        one_line, cf0_mean = create_data_line(wav, speaker_id, d, hubert)
        if counter % VAL_PER != 0:
            output_file_list.append(one_line)
        else:
            output_file_list_val.append(one_line)
        counter = counter +1
        cf0_mean_t = cf0_mean_t + cf0_mean
    cf0_mean_t = cf0_mean_t / counter
    Correspondence_list.append(str(speaker_id)+"|"+ str(cf0_mean_t) + "|" + os.path.basename(d) + "\n")

    lists = [output_file_list, output_file_list_val, Correspondence_list]
    write_configs(lists, filename)
    return NUM_MAX_SID

def create_dataset_character(filename, tid):
    #list回りの宣言とソート
    textful_dir_list = glob.glob("dataset/**/")
    textful_dir_list.sort()
    Correspondence_list = list()
    output_file_list = list()
    output_file_list_val = list()
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")
    #paths
    my_path = "dataset/00_myvoice"
    target_path = "dataset/01_target"

    #set list wav and text
    #myvoice
    speaker_id = MY_SID
    d = my_path
    wav_file_list = glob.glob(d + "/wav/*.wav")
    wav_file_list.sort()
    if len(wav_file_list) == 0:
        print("Error" + d + "/wav に音声データがありません")
        exit()
    counter = 0
    cf0_mean_t = 0.0
    for wav in wav_file_list:
        one_line, cf0_mean = create_data_line(wav, speaker_id, d, hubert)
        if counter % VAL_PER != 0:
            output_file_list.append(one_line)
        else:
            output_file_list_val.append(one_line)
        counter = counter +1
        cf0_mean_t = cf0_mean_t + cf0_mean
    cf0_mean_t = cf0_mean_t / counter
    Correspondence_list.append(str(speaker_id)+"|"+ str(cf0_mean_t) + "|" + os.path.basename(d) + "\n")

    speaker_id = tid
    d = target_path
    wav_file_list = glob.glob(d + "/wav/*.wav")
    wav_file_list.sort()
    if len(wav_file_list) == 0:
        print("Error" + d + "/wav に音声データがありません")
        exit()
    counter = 0
    cf0_mean_t = 0.0
    for wav in wav_file_list:
        one_line, cf0_mean = create_data_line(wav, speaker_id, d, hubert)
        if counter % VAL_PER != 0:
            output_file_list.append(one_line)
        else:
            output_file_list_val.append(one_line)
        counter = counter +1
        cf0_mean_t = cf0_mean_t + cf0_mean
    cf0_mean_t = cf0_mean_t / counter
    Correspondence_list.append(str(speaker_id)+"|"+ str(cf0_mean_t) + "|" + os.path.basename(d) + "\n")

    lists = [output_file_list, output_file_list_val, Correspondence_list]
    write_configs(lists, filename)
    return NUM_MAX_SID

def create_dataset_multi_character(filename, file_path):
    #list回りの宣言とソート
    textful_dir_list = glob.glob("dataset/**/")
    textful_dir_list.sort()
    Correspondence_list = list()
    output_file_list = list()
    output_file_list_val = list()
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")

    with open(file_path, "r") as f:
        for line in f.readlines():
            target_dir , sid = line.split("|")
            sid = sid.rstrip('\n')
            wav_file_list = glob.glob("dataset/" + target_dir + "/wav/*.wav")
            wav_file_list.sort()
            if len(wav_file_list) == 0:
                print("Error" + target_dir + "/wav に音声データがありません")
                exit()
            counter = 0
            cf0_mean_t = 0.0
            for wav in wav_file_list:
                one_line, cf0_mean = create_data_line(wav, sid, target_dir, hubert)
                if counter % VAL_PER != 0:
                    output_file_list.append(one_line)
                else:
                    output_file_list_val.append(one_line)
                counter = counter +1
                cf0_mean_t = cf0_mean_t + cf0_mean
            cf0_mean_t = cf0_mean_t / counter
            Correspondence_list.append(str(sid)+"|"+ str(cf0_mean_t) +"|"+ target_dir + "\n")

    lists = [output_file_list, output_file_list_val, Correspondence_list]
    write_configs(lists, filename)
    return NUM_MAX_SID

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, required=True,
                        help='filelist for configuration')
    parser.add_argument('-s', '--sr', type=int, default=24000,
                        help='sampling rate (default = 24000)')
    parser.add_argument('-t', '--target', type=int, default=9999,
                        help='pre_traind targetid (zundamon = 101, sora = 102, methane = 103, tsumugi = 104)')
    parser.add_argument('-m', '--multi_target', type=str, default=None,
                        help='pre_traind targetid (zundamon = 101, sora = 102, methane = 103, tsumugi = 104)')
    parser.add_argument('-c', '--config', type=str, default="./configs/baseconfig.json",
                        help='JSON file for configuration')
    args = parser.parse_args()
    filename = args.filename
    print(filename)

    if args.multi_target != None:
        n_spk = create_dataset_multi_character(filename, args.multi_target)
    elif args.target != 9999 and args.target == ZUNDAMON_SID:
        n_spk = create_dataset_zundamon(filename)
    elif args.target != 9999:
        n_spk = create_dataset_character(filename, args.target)
    else:
        n_spk = create_dataset(filename)
    
    create_json(filename, n_spk, args.sr, args.config)

if __name__ == '__main__':
    main()
