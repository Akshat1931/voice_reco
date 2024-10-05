import os
import pyaudio
import wave

def get_next_filename(directory):
    """
    Generate a filename based on the existing files in the directory.
    Filename format: speaker_sample_<number>.wav
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # Filter for .wav files
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    
    # Extract the numbers from the filenames
    existing_numbers = []
    for f in wav_files:
        try:
            number = int(f.split('_')[2].replace('.wav', ''))
            existing_numbers.append(number)
        except (IndexError, ValueError):
            continue
    
    # Generate the next number
    next_number = max(existing_numbers) + 1 if existing_numbers else 1
    
    return os.path.join(directory, f'speaker_sample_{next_number}.wav')

def record_audio(directory, duration=5):
    """
    Records audio and saves it as a .wav file in the specified directory.
    """
    chunk = 1024  # Number of frames per buffer
    sample_format = pyaudio.paInt16  # 16-bit audio format
    channels = 1  # Mono audio
    fs = 16000  # Sampling rate (16kHz)
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    print('Recording')

    # Open stream for audio input
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []

    # Read and append data to frames
    for _ in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    print('Finished recording')

    # Ensure directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save recorded audio
    filename = get_next_filename(directory)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
    
    print(f'Audio saved as {filename}')

if __name__ == "__main__":
    record_audio('dataset')
