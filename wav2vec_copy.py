#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pyaudio
import wave
import threading
import random
import time
import csv
import subprocess
from fcsn_l import FCSN4L

import argparse
import os
import os.path as osp
import tqdm
import torch
import torch.nn.functional as F
from shutil import copyfile

from npy_append_array import NpyAppendArray

import fairseq
import soundfile as sf


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute kmeans codebook from kaldi-computed feats"
    )
    # fmt: off
    parser.add_argument('data', help='location of tsv files')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--checkpoint', type=str, help='checkpoint for wav2vec ctc model', required=True)
    parser.add_argument('--layer', type=int, default=14, help='which layer to use')
    # fmt: on

    return parser


class Wav2VecFeatureReader(object):
    def __init__(self, cp_file, layer):
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_file]
        )
        model = model[0]
        model.eval()
        model.cuda()
        self.model = model
        self.task = task
        self.layer = layer

    def read_audio(self, fname):
        """Load an audio file and return PCM along with the sample rate"""
        wav, sr = sf.read(fname)
        assert sr == 16e3

        return wav

    def get_feats(self, loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                assert source.dim() == 1, source.dim()
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)

            m_res = self.model(source=source, mask=False, features_only=True, layer=self.layer)
            return m_res["x"].squeeze(0).cpu()


def get_iterator(args, reader):
    with open(osp.join(args.data, args.split) + ".tsv", "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        files = [osp.join(root, line.split("\t")[0]) for line in lines if len(line) > 0]

        num = len(files)
        #reader = Wav2VecFeatureReader(args.checkpoint, args.layer)

        def iterate():
            for fname in files:
                w2v_feats = reader.get_feats(fname)
                print("features extracted from file: ", fname)
                yield w2v_feats

    return iterate, num

### Our addition starts here ###

# def run_wav2vec(wav_path): 
#     sh_path = '/home/ozkan/Desktop/ELEC491/prepare_audio_v3.sh'
#     embedding_path = '/home/ozkan/Desktop/ELEC491/out'   
#     model_path = '/home/ozkan/Desktop/ELEC491/fairseq/wav2vec_vox_new.pt'
#     #subprocess.run(f'zsh {sh_path} {wav_path} {embedding_path} {model_path} 128 14')
#     subprocess.run(["zsh", sh_path, wav_path, embedding_path, model_path, "128", "14"])
    
def remove_files(files, path):
    for file in files:
        #copyfile(f'{path}{file}', f'/home/ozkan/Desktop/ELEC491/saved/{file}')
        os.remove(f'{path}{file}')
        print(f"Consumed the wave file with name {file}")

def create_tsv(ready_files, path):
    # prepare train.tsv
    
    p = f'{path}test.tsv'  # Modified from train to test
    with open(p, 'wt') as out_file: 
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow([path])
        for file in ready_files:
            tsv_writer.writerow([file])
    

def return_ready_files(path):   
    
    # Get a list of all .wav files in the folder
    wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]
    ready_files = []

    # Iterate over the .wav files
    for wav_file in wav_files:
        file = f'{path}{wav_file}'
        try:
            os.rename(file, file)
            ready_files.append(wav_file)

        # Skip processing if file is in still in use by another process
        except PermissionError:
            break
    ready_files.sort()
    return ready_files
    

def consume_wav_files(path, stop, args, reader, model):
    # Continuously look for new .wav files in the folder
    while True:
    
        try:
            ready_files = return_ready_files(path)
            if len(ready_files) > 0:
                create_tsv(ready_files, path)
                #run_wav2vec(path)
                iterator, num = extract_features(args, reader)
                process_features(model, iterator, num)
                remove_files(ready_files, path)
            else:
                # Sleep to reduce CPU usage
                time.sleep(random.random())    

        # Skip processing if file is in still in use by another process
        except PermissionError:  
            print("race cond")
            break

        # Stop consumer thread
        if stop():
            break

def create_files(args, dest):
        copyfile(osp.join(args.data, args.split) + ".tsv", dest + ".tsv")

        if osp.exists(dest + ".npy"):
            os.remove(dest + ".npy")
        npaa = NpyAppendArray(dest + ".npy")
        return npaa

def extract_features(args, reader):
    #save_path = osp.join(args.save_dir, args.split)
    #npaa = create_files(args, save_path)

    generator, num = get_iterator(args, reader)
    iterator = generator()

    return iterator, num
    #for w2v_feats in tqdm.tqdm(iterator, total=num):
    #    print("w2v_feats shape:", w2v_feats.shape)

def process_features(model, iterator, num):
    for w2v_feats in tqdm.tqdm(iterator, total=num):
        print("w2v_feats shape:", w2v_feats.shape)
        predictions = predict_labels(model, w2v_feats)


def predict_labels(model, features):
    
    # Assume data length is small
    X = torch.stack([features[0:150].unsqueeze(0), features[50:200].unsqueeze(0), features[99:249].unsqueeze(0)])

    with torch.no_grad():
        output = model.forward(0, X)
        #print(output)
        predictions = [torch.argmax(output[i]).item() for i in range(len(X))]
    # for pred in predictions:
    #     print("Predicted label: ", pred)
    print("Predicted labels: ", predictions[0], predictions[1], predictions[2])
    return predictions

def main():

    ##### Our variables

    # Set the parameters of the audio recording
    CHUNK = 1000
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5

    #OUT_PATH = "C:/Users/VCA/Desktop/Elec491/recording_test/"
    OUT_PATH = "/home/ozkan/Desktop/ELEC491/recording_test/"
    MODEL_PATH = "/home/ozkan/Desktop/ELEC491/model_state.pt"
    STOP_THREADS = False

    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    #Initially loading w2v model
    reader = Wav2VecFeatureReader(args.checkpoint, args.layer)

    model = FCSN4L(2, 1)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Create a PyAudio object
    p = pyaudio.PyAudio()

    # Open a stream for audio recording
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")

    # Start the consumer thread
    consumer_thread = threading.Thread(target=consume_wav_files, args=(OUT_PATH, lambda : STOP_THREADS, args, reader, model))
    consumer_thread.start()

    # Start recording in a loop
    wav_id = 0
    while True:
        try:
            # Record n-second audio
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            # Save the recorded audio to a wave file
            formatted_id = "{:04d}".format(wav_id) # 5 --> 0005
            file_name = f'{OUT_PATH}recorded_{formatted_id}.wav'
            with wave.open(file_name, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))

            wav_id += 1

        except KeyboardInterrupt:
            # Stop the stream and close it
            stream.stop_stream()
            stream.close()

            # Close the PyAudio object
            p.terminate()

            # Stop the consumer thread
            STOP_THREADS = True
            consumer_thread.join()

            # Exit the loop
            break


if __name__ == "__main__":
    main()
