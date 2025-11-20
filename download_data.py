import os
import torchaudio

def download_librispeech(root="./data"):
    if not os.path.exists(root):
        os.makedirs(root)
    
    print("Downloading LibriSpeech train-clean-360 (approx 23GB)...")
    print("This may take a while. Go grab a coffee ☕")
    
    # This contains ~920 speakers and 360 hours of speech
    torchaudio.datasets.LIBRISPEECH(root=root, url="train-clean-360", download=True)
    
    # OPTIONAL: Keep dev-clean for validation if you already downloaded it
    # torchaudio.datasets.LIBRISPEECH(root=root, url="dev-clean", download=True)

if __name__ == "__main__":
    download_librispeech()