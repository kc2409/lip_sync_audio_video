import cv2
import numpy as np
import torch
from Wav2Lip.models import Wav2Lip
from Wav2Lip.preprocess import process_video_file
import librosa
import argparse
import moviepy.editor as mp
import os

# Set the path to the pre-trained Wav2Lip model
checkpoint_path = "wav2lip.pth"

# Set the paths to the video and audio files
video_path = "test copy.mp4"
audio_path = "test1 copy.wav"

# Set the output video path
output_path = "output_video.mp4"

# Load the pre-trained Wav2Lip model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Wav2Lip().to(device)

# Load the model's state_dict
#model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])

model.eval()

args = argparse.Namespace(batch_size=4, preprocessed_root="result")
gpu_id = 0

# Set the desired new width and height for the video
new_width = 640
new_height = 480

# Load the video and audio files
video = mp.VideoFileClip(video_path)
audio = mp.AudioFileClip(audio_path)

# Get durations of video and audio
video_duration = video.duration
audio_duration = audio.duration

# Check if video duration is longer than audio duration
if video_duration > audio_duration:
    # Truncate the video to match the audio duration
    video = video.subclip(0, audio_duration)
else:
    # Loop the audio to match the video duration
    audio = audio.loop(duration=video_duration)

# Set the audio of the video
video = video.set_audio(audio)

# Export the synchronized video with audio
video.write_videofile(output_path, codec="libx264")

# Remove the temporary lip-synced video file
os.remove(output_path)

# Extract frames from the video and resize
frames, fps = process_video_file(output_path, args, gpu_id, new_width, new_height)

# Load the audio file
audio, sr = librosa.load(audio_path, sr=44100)

# Synchronize lip movements
output_frames = []
for i in range(len(frames)):
    # Preprocess the frame and audio
    frame = frames[i]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)  # Convert to float

    audio_clip = audio[int(i * fps):int((i + 1) * fps)]  # Extract audio clip for the corresponding frame
    mel = librosa.feature.melspectrogram(y=audio_clip, sr=sr, n_fft=1024, hop_length=256)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).float().to(device)  # Convert to float

    # Perform lip-syncing
    with torch.no_grad():
        result = model(audio_sequences=mel.to(device), face_sequences=frame.to(device))  # Move tensors to device

    # Convert the tensor output back to an image
    output_frame = result[0].data.cpu().numpy().transpose(1, 2, 0)
    output_frame = (255 * output_frame).astype(np.uint8)
    output_frames.append(output_frame)

# Save the lip-synced frames as a video
output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_width, new_height))

for frame in output_frames:
    output_video.write(frame)

output_video.release()

video.write_videofile(output_path, codec="libx264")

print("Lip-syncing completed. Output saved as", output_path)