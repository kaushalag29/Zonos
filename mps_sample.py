import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

# torch._dynamo.config.suppress_errors = True

# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the model and move it to the MPS device
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

# Load audio file
wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")

# Move audio tensor to the MPS device
wav = wav.to(device)

# Generate speaker embedding and move it to the MPS device
speaker = model.make_speaker_embedding(wav, sampling_rate).to(device)

# Set random seed for reproducibility
torch.manual_seed(421)

# Create conditioning dictionary
cond_dict = make_cond_dict(text="Hello, world!", speaker=speaker, language="en-us")

# Function to move tensors to the specified device
def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    elif isinstance(value, (list, tuple)):
        return [move_to_device(v, device) if isinstance(v, torch.Tensor) else v for v in value]
    elif isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    else:
        return value

# Move all tensors in cond_dict to the MPS device
cond_dict = {key: move_to_device(value, device) for key, value in cond_dict.items()}

# Prepare conditioning and move it to the MPS device
conditioning = model.prepare_conditioning(cond_dict).to(device)

# Generate codes and move them to the MPS device
codes = model.generate(conditioning).to(device)

# Decode the codes to audio and move the result back to the CPU for saving
wavs = model.autoencoder.decode(codes).cpu()

# Save the generated audio
torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)
