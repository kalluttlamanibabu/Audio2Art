import torch
import whisper
import IPython.display as display
import speech_recognition as sr
from diffusers import StableDiffusionPipeline
from google.colab import output
import pydub
import os
from google.colab import output
import base64
import wave
import io

# JavaScript code for recording
RECORD_JS = """
const sleep  = time => new Promise(resolve => setTimeout(resolve, time));
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader();
  reader.onloadend = () => resolve(reader.result);
  reader.readAsDataURL(blob);
});
async function record(sec) {
  stream = await navigator.mediaDevices.getUserMedia({audio:true});
  rec = new MediaRecorder(stream);
  chunks = [];
  rec.ondataavailable = e => chunks.push(e.data);
  rec.start();
  await sleep(sec * 1000);
  rec.stop();
  await sleep(1000);
  blob = new Blob(chunks);
  text = await b2text(blob);
  return text;
}
"""

# Run JavaScript in Colab before calling the record_audio function
output.eval_js(RECORD_JS)

def record_audio(file_path="input.wav", duration=5):
    print(f"Recording... Speak for {duration} seconds!")

    # Call the JavaScript function
    audio_b64 = output.eval_js(f"record({duration})").split(",")[1]

    # Convert base64 to WAV format
    audio_bytes = base64.b64decode(audio_b64)

    # Save to a .wav file
    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    print(f"Recording saved as {file_path}")
    return file_path

# Run this to start recording
record_audio()
import IPython.display as display

def play_audio(file_path="input.wav"):
    print("Playing recorded audio...")
    display.display(display.Audio(file_path))

# Play the recorded audio
play_audio()
from google.colab import output
import base64
import wave
import io

# JavaScript code for recording
RECORD_JS = """
const sleep  = time => new Promise(resolve => setTimeout(resolve, time));
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader();
  reader.onloadend = () => resolve(reader.result);
  reader.readAsDataURL(blob);
});
async function record(sec) {
  stream = await navigator.mediaDevices.getUserMedia({audio:true});
  rec = new MediaRecorder(stream);
  chunks = [];
  rec.ondataavailable = e => chunks.push(e.data);
  rec.start();
  await sleep(sec * 1000);
  rec.stop();
  await sleep(1000);
  blob = new Blob(chunks);
  text = await b2text(blob);
  return text;
}
"""

# Run JavaScript in Colab
output.eval_js(RECORD_JS)
from google.colab import output
import base64
import wave
import io

# JavaScript code for recording
RECORD_JS = """
const sleep  = time => new Promise(resolve => setTimeout(resolve, time));
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader();
  reader.onloadend = () => resolve(reader.result);
  reader.readAsDataURL(blob);
});
async function record(sec) {
  stream = await navigator.mediaDevices.getUserMedia({audio:true});
  rec = new MediaRecorder(stream);
  chunks = [];
  rec.ondataavailable = e => chunks.push(e.data);
  rec.start();
  await sleep(sec * 1000);
  rec.stop();
  await sleep(1000);
  blob = new Blob(chunks);
  text = await b2text(blob);
  return text;
}
"""

def record_audio(file_path="input.wav", duration=5):
    # Run JavaScript in Colab before calling the record function
    output.eval_js(RECORD_JS) # This ensures the JavaScript code is executed

    print(f"Recording... Speak for {duration} seconds!")

    # Call the JavaScript function
    audio_b64 = output.eval_js(f"record({duration})").split(",")[1]

    # Convert base64 to WAV format
    audio_bytes = base64.b64decode(audio_b64)

    # Save to a .wav file
    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    print(f"Recording saved as {file_path}")
    return file_path

# Run this to start recording
record_audio()
import IPython.display as display

def play_audio(file_path="input.wav"):
    print("Playing recorded audio...")
    display.display(display.Audio(file_path))

# Play the recorded audio
play_audio()
import whisper

def transcribe_audio(audio_path):
    print("Loading Whisper model...")
    model = whisper.load_model("base")  # Change to "small", "medium", or "large" if needed
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    print("Transcription:", result["text"])
    return result["text"]

# Transcribe the recorded audio
transcribed_text = transcribe_audio("input.wav")
from diffusers import StableDiffusionPipeline
import torch

def generate_image(prompt):
    print("Loading Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    print(f"Generating image for prompt: {prompt}")
    image = pipe(prompt).images[0]
    image.save("generated_image.png")
    print("Image saved as generated_image.png")
    return image

# Generate the AI image
generated_image = generate_image(transcribed_text)

# Display the generated image in Colab
display.display(generated_image)
def transcribe_audio(audio_path):
    print("Loading Whisper model...")
    model = whisper.load_model("base")  # You can change to "small" or "large"
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    print("Transcription:", result["text"])
    return result["text"]

# Transcribe the recorded audio
transcribed_text = transcribe_audio("input.wav")

    image.save("generated_image.png")
    print("Image saved as generated_image.png")
    return image

# Generate the AI image
generated_image = generate_image(transcribed_tedef generate_image(prompt):
    print("Loading Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    print(f"Generating image for prompt: {prompt}")
    image = pipe(prompt).images[0]xt)

# Display the generated image in Colab
display.display(generated_image)
