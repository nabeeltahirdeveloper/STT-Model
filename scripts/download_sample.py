import os

import soundfile as sf
from datasets import load_dataset

print("Connecting to Hugging Face...")
# We use streaming=True so we don't download the whole massive dataset, just the first file!
ds = load_dataset(
    "mozilla-foundation/common_voice_17_0",
    "ur",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

# Grab the very first audio clip
sample = next(iter(ds))

# Create the folder the test script wants
os.makedirs("data/eval/alignment", exist_ok=True)

# Save the audio as a .wav file
audio_array = sample["audio"]["array"]
sample_rate = sample["audio"]["sampling_rate"]
sf.write("data/eval/alignment/sample_01.wav", audio_array, sample_rate)

print("Success! Downloaded a free Urdu audio clip to data/eval/alignment/sample_01.wav")
print("The original person said (in Urdu script):", sample["sentence"])
