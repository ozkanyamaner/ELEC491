import pyaudio
import wave
import os
import threading
import random
import time
from shutil import copyfile

def remove_file(file):
    os.remove(file)
    print(f"Consumed the wave file with name {file}")

def consume_wav_files(path, stop):
    # Continuously look for new .wav files in the folder
    while not STOP_THREADS:
        # Get a list of all .wav files in the folder
        wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]

        # Iterate over the .wav files
        for wav_file in wav_files:
            file = f'{path}{wav_file}'
            # try:
            #     remove_file(file)

            # # Skip processing if file is in still in use by another process
            # except PermissionError:  
            #     print("race cond")
            #     break

        # Stop consumer thread
        if stop():
            break

        # Sleep to reduce CPU usage
        time.sleep(random.random())


# Set the parameters of the audio recording
CHUNK = 1000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3

OUT_PATH = "C:/Users/VCA/Desktop/Elec491/recording_test/"
IN_PATH = "C:/Users/VCA/Desktop/Elec491/input_wavs/"
COPY_PATH = "C:/Users/VCA/Desktop/Elec491/saved/"
STOP_THREADS = False

# Create a PyAudio object
p = pyaudio.PyAudio()

# Open a stream for audio recording
# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=CHUNK)

out_stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
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

            for iter in range(num_frames // (RATE * sample_width)):

                # Dummy loop
                # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                #     dummy = stream.read(CHUNK)
                
                start = iter * frame_per_recording
                end = (iter + 1) * frame_per_recording
                out_stream.write(frames[start:end])

                formatted_id = "{:04d}".format(wav_id) # 5 --> 0005
                file_name = f'{OUT_PATH}recorded_{formatted_id}.wav'
                with wave.open(file_name, "wb") as wf_out:
                    wf_out.setnchannels(CHANNELS)
                    wf_out.setsampwidth(p.get_sample_size(FORMAT))
                    wf_out.setframerate(RATE)
                    wf_out.writeframes(frames[start:end])

                wav_id += 1
            
            # copyfile(f'{IN_PATH}{file}', f'{COPY_PATH}{file}')
            # os.remove(f'{IN_PATH}{file}')

        # Went through all files
        print("Went through all files.")



    except KeyboardInterrupt:
        # Stop the stream and close it
        # stream.stop_stream()
        # stream.close()
        out_stream.stop_stream()
        out_stream.close()
        # Close the PyAudio object
        p.terminate()

        # Stop the consumer thread
        STOP_THREADS = True
        consumer_thread.join()

        # Exit the loop
        break

# stream.stop_stream()
# stream.close()
out_stream.stop_stream()
out_stream.close()
p.terminate()
STOP_THREADS = True
consumer_thread.join()

