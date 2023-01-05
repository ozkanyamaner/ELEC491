import pyaudio
import wave
import os
import threading
import random
import time
import csv
import subprocess

def run_wav2vec(wav_path): 
    sh_path = '/home/ozkan/Desktop/ELEC491/prepare_audio_v3.sh'
    embedding_path = '/home/ozkan/Desktop/ELEC491/out'   
    model_path = '/home/ozkan/Desktop/ELEC491/fairseq/wav2vec_vox_new.pt'
    #subprocess.run(f'zsh {sh_path} {wav_path} {embedding_path} {model_path} 128 14')
    subprocess.run(["zsh", sh_path, wav_path, embedding_path, model_path, "128", "14"])
    
def remove_files(files, path):
    for file in files:
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
    

def consume_wav_files(path, stop):
    # Continuously look for new .wav files in the folder
    while not STOP_THREADS:
    
        try:
            ready_files = return_ready_files(path)
            if len(ready_files) > 0:
                create_tsv(ready_files, path)
                run_wav2vec(path)
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

        


# Set the parameters of the audio recording
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3

#OUT_PATH = "C:/Users/VCA/Desktop/Elec491/recording_test/"
OUT_PATH = "/home/ozkan/Desktop/ELEC491/recording_test/"
STOP_THREADS = False

# Change the working directory to the folder where this script is located
os.chdir('/home/ozkan/Desktop/ELEC491/fairseq/')

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
consumer_thread = threading.Thread(target=consume_wav_files, args=(OUT_PATH, lambda : STOP_THREADS))
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
