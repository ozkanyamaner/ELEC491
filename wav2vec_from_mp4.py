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

import fairseq
import soundfile as sf
from furhat_remote_api import FurhatRemoteAPI


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
        copyfile(f'{path}{file}', f'/home/ozkan/Desktop/ELEC491/saved/{file}')
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
    
def generate_response(furhat, predictions):
    if 1 in predictions:
        furhat.gesture(name="Surprise")
        furhat.say(url="file:///home/ozkan/Desktop/ELEC491/bellring_short.wav", lipsync=False)

def play_video(filename):
        opener = "xdg-open"               
        subprocess.call([opener, filename])

def consume_wav_files(path, stop, args, reader, model, furhat):

    
    # Continuously look for new .wav files in the folder
    rest = None # Unprocessed segment of the wav file
    shift = 10 # Amount of shift in windows during processing
    runtime = 3 - shift / 50

    while True:
    
        try:
            ready_files = return_ready_files(path)
            if len(ready_files) > 0:
                create_tsv(ready_files, path)
                #run_wav2vec(path)
                iterator, num = extract_features(args, reader)
                predictions, rest = process_features(model, iterator, num, rest, shift)
                generate_response(furhat, predictions)
                runtime += len(predictions) * shift / 50 # shift / 50 is time required for 1 prediction
                remove_files(ready_files, path)
                print(f'Latest response: {runtime}')
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


def extract_features(args, reader):

    generator, num = get_iterator(args, reader)
    iterator = generator()

    return iterator, num
    #for w2v_feats in tqdm.tqdm(iterator, total=num):
    #    print("w2v_feats shape:", w2v_feats.shape)


def process_features(model, iterator, num, rest, shift):
    all_predictions = []
    # for w2v_feats in tqdm.tqdm(iterator, total=num):
    for w2v_feats in iterator:
        if rest is not None:
            w2v_feats = torch.cat((rest, w2v_feats))

        if len(w2v_feats) < 150: # cumulate nonprocessed data
                rest = w2v_feats
                continue
        # print("w2v_feats shape:", w2v_feats.shape)
        predictions, rest = predict_labels(model, w2v_feats, shift)
        all_predictions += predictions
    
    
    return all_predictions, rest


def predict_labels(model, features, shift):
    
    # Assume data length is small
    X, rest = get_overlapping_features(features, shift)

    with torch.no_grad():
        output = model.forward(0, X)
        predictions = [torch.argmax(output[i]).item() for i in range(len(X))]
    print(f'Predicted labels: {predictions}')
    return predictions, rest

def get_overlapping_features(features, shift):
    chunk = 150
    length = len(features)
    start, end = 0, chunk
    lst = []
    while end < length:
        lst.append(features[start:end].unsqueeze(0))
        start += shift
        end += shift
    
    lst.append(features[length - chunk:length].unsqueeze(0))
    rest = features[length - chunk + shift:length]
    return torch.stack(lst), rest

def main():

    input_mic = False
    with_mp4 = True
    furhat_virtual = True 
    ##### Our variables
    # Set the parameters of the audio recording
    CHUNK = 1000
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 1

    IN_PATH = "/home/ozkan/Desktop/ELEC491/wavs/"
    OUT_PATH = "/home/ozkan/Desktop/ELEC491/recording_test/"
    MODEL_PATH = "/home/ozkan/Desktop/ELEC491/model_state_with_5_1.pt"
    MP4_PATH = "/home/ozkan/Desktop/ELEC491/MP4s/"
    STOP_THREADS = False

    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    if furhat_virtual:
        furhat = FurhatRemoteAPI("localhost")
    else:
        furhat = FurhatRemoteAPI("192.168.137.1")

    furhat.gesture(name="CloseEyes")

    #Initially loading w2v model
    reader = Wav2VecFeatureReader(args.checkpoint, args.layer)

    model = FCSN4L(2, 1)
    model.load_state_dict(torch.load(MODEL_PATH)) # Loading model to cpu or gpu
    #model.to('cpu') ######
    model.eval()

    

    user_input = input("Press enter to continue: ")
    furhat.gesture(name="OpenEyes")

    # Create a PyAudio object
    p = pyaudio.PyAudio()

    # Open a stream for audio recording
    if input_mic:
        print("Input type: microphone")
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    elif with_mp4:
        print("Input type: mp4 file")
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    else:
        print("Input type: wav file")
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    # Start the consumer thread
    consumer_thread = threading.Thread(target=consume_wav_files, args=(OUT_PATH, lambda : STOP_THREADS, args, reader, model, furhat))
    consumer_thread.start()

    # Start recording in a loop
    wav_id = 0
    while True:
        try:
            if input_mic:
                
                # Record n-second audio
                frames = []
                for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    data = stream.read(CHUNK)
                    frames.append(data)

                # Save the recorded audio to a wave file
                formatted_id = "{:06d}".format(wav_id) # 5 --> 000005
                file_name = f'{OUT_PATH}recorded_{formatted_id}.wav'
                with wave.open(file_name, "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))

                wav_id += 1
                print(f'Recording duration: {wav_id * RECORD_SECONDS}')

            else:
                
                # Record n-second audio
                for file in os.listdir(IN_PATH):
                    if not file.endswith('.wav'): continue

                    full_name = f'{IN_PATH}{file}'
                    with wave.open(full_name, 'rb') as wf_in:
                        assert wf_in.getnchannels() == CHANNELS
                        assert wf_in.getframerate() == RATE
                        sample_width = wf_in.getsampwidth()
                        frames = wf_in.readframes(wf_in.getnframes())
                        num_frames = len(frames)
                        frame_per_recording = RECORD_SECONDS * RATE * sample_width

                    print("number of iterations:", num_frames // frame_per_recording)
                    if with_mp4:
                        try:
                            play_video(f'{MP4_PATH}{file[:-4]}.mp4')
                        except:
                            print("No mp4 file found")
                            continue
                    for iter in range(int(num_frames // frame_per_recording)):

                        start = iter * frame_per_recording
                        end = (iter + 1) * frame_per_recording
                        if with_mp4:    
                            time.sleep(RECORD_SECONDS)
                        else:
                            stream.write(frames[int(start):int(end)])    

                        formatted_id = "{:06d}".format(wav_id) # 5 --> 000005
                        file_name = f'{OUT_PATH}wav_audio_{formatted_id}.wav'
                        with wave.open(file_name, "wb") as wf_out:
                            wf_out.setnchannels(CHANNELS)
                            wf_out.setsampwidth(p.get_sample_size(FORMAT))
                            wf_out.setframerate(RATE)
                            wf_out.writeframes(frames[int(start):int(end)])

                        wav_id += 1

                    # copyfile(f'{IN_PATH}{file}', f'{COPY_PATH}{file}')
                    # os.remove(f'{IN_PATH}{file}')

                # Went through all files
                print("Went through all files.")
            
        except KeyboardInterrupt:
            # Stop the stream and close it
            stream.stop_stream()
            stream.close()

            # Close the PyAudio object
            p.terminate()

            # Stop the consumer thread
            STOP_THREADS = True
            consumer_thread.join()

            for file in os.listdir(OUT_PATH):
                if file.endswith('.wav'):
                    os.remove(f'{OUT_PATH + file}')

            # Exit the loop
            break

    # Stop the stream and close it
    stream.stop_stream()
    stream.close()

    # Close the PyAudio object
    p.terminate()

    # Stop the consumer thread
    STOP_THREADS = True
    consumer_thread.join()

    for file in os.listdir(OUT_PATH):
        if file.endswith('.wav'):
            os.remove(f'{OUT_PATH + file}')
    # Exit the loop    

if __name__ == "__main__":
    main()
