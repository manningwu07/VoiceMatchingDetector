# scripts/download_common_voice.py
import os
import subprocess
from pathlib import Path
import json

def download_common_voice(output_dir="data/raw/common_voice", target_size_mb=300, lang="en"):
    """
    Download Common Voice dataset (~200-300MB).
    
    Args:
        output_dir: Where to save
        target_size_mb: Approximate target size
        lang: Language code (en, fr, de, etc.)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Common Voice ({lang}) to {output_dir}")
    print(f"Target size: ~{target_size_mb}MB")
    
    # Use huggingface_hub to download (easiest)
    try:
        from huggingface_hub import hf_hub_download
        print("Using huggingface_hub...")
        
        # Common Voice is on HuggingFace
        repo_id = "mozilla-foundation/common_voice_13_0"
        
        # Download a smaller subset (train-0 is usually ~300-500MB)
        filename = f"{lang}/train.tar"
        
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(output_dir),
            force_download=False,
        )
        print(f"✅ Downloaded to {output_dir}")
        
    except ImportError:
        print("huggingface_hub not found, trying wget...")
        download_with_wget(output_dir, lang, target_size_mb)

def download_with_wget(output_dir, lang="en", target_size_mb=300):
    """Fallback: direct wget from Common Voice downloads."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Direct links to Common Voice 13.0 tar files
    # These are ~300-500MB each per language
    urls = {
        "en": "https://downloads.commonvoice.mozilla.org/cv-corpus-13.0/cv-corpus-13.0-2023-11-09/en.tar.gz",
        "fr": "https://downloads.commonvoice.mozilla.org/cv-corpus-13.0/cv-corpus-13.0-2023-11-09/fr.tar.gz",
        "de": "https://downloads.commonvoice.mozilla.org/cv-corpus-13.0/cv-corpus-13.0-2023-11-09/de.tar.gz",
    }
    
    url = urls.get(lang)
    if not url:
        print(f"Language {lang} not in hardcoded URLs; check https://commonvoice.mozilla.org/")
        return
    
    print(f"Downloading {lang} from {url}...")
    tar_path = output_dir / f"{lang}.tar.gz"
    
    # wget with resume
    subprocess.run([
        "wget", "-c", "-O", str(tar_path), url
    ], check=True)
    
    print(f"✅ Downloaded to {tar_path}")
    print(f"Extracting...")
    
    subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(output_dir)], check=True)
    print(f"✅ Extracted to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw/common_voice")
    parser.add_argument("--lang", default="en", help="Language code")
    parser.add_argument("--size-mb", type=int, default=300, help="Target size in MB")
    
    args = parser.parse_args()
    
    download_common_voice(
        output_dir=args.output_dir,
        target_size_mb=args.size_mb,
        lang=args.lang,
    )