# scripts/prepare_pairs.py
import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
import random
import argparse

def prepare_pairs(audio_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Organize audio by speaker, generate positive/negative pairs.
    
    Expected structure:
        audio_dir/
            speaker_001/
                utterance_001.wav
                utterance_002.wav
            speaker_002/
                ...
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Collect speakers and utterances
    speakers = defaultdict(list)
    for speaker_dir in os.listdir(audio_dir):
        speaker_path = os.path.join(audio_dir, speaker_dir)
        if not os.path.isdir(speaker_path):
            continue
        wavs = [
            os.path.join(speaker_path, f)
            for f in os.listdir(speaker_path)
            if f.endswith(".wav")
        ]
        speakers[speaker_dir] = wavs
    
    print(f"Found {len(speakers)} speakers, {sum(len(v) for v in speakers.values())} utterances")
    
    # Split speakers into train/val/test
    speaker_list = list(speakers.keys())
    random.shuffle(speaker_list)
    n_train = int(len(speaker_list) * train_ratio)
    n_val = int(len(speaker_list) * val_ratio)
    
    train_speakers = speaker_list[:n_train]
    val_speakers = speaker_list[n_train:n_train + n_val]
    test_speakers = speaker_list[n_train + n_val:]
    
    splits = {
        "train": train_speakers,
        "val": val_speakers,
        "test": test_speakers,
    }
    
    # Generate pairs for each split
    for split_name, spk_list in splits.items():
        pairs = []
        
        # Positive pairs (same speaker)
        for spk in spk_list:
            wavs = speakers[spk]
            if len(wavs) >= 2:
                for i in range(len(wavs)):
                    for j in range(i + 1, len(wavs)):
                        pairs.append({
                            "audio_1": wavs[i],
                            "audio_2": wavs[j],
                            "label": 1,
                        })
        
        # Negative pairs (different speaker)
        # For each speaker, pick random negatives from other speakers
        for spk in spk_list:
            wavs_pos = speakers[spk]
            other_spks = [s for s in spk_list if s != spk]
            for wav_pos in wavs_pos:
                for _ in range(2):  # 2 negatives per positive utterance
                    other_spk = random.choice(other_spks)
                    wav_neg = random.choice(speakers[other_spk])
                    pairs.append({
                        "audio_1": wav_pos,
                        "audio_2": wav_neg,
                        "label": 0,
                    })
        
        df = pd.DataFrame(pairs)
        csv_path = output_dir / f"{split_name}_pairs.csv"
        df.to_csv(csv_path, index=False)
        print(f"{split_name}: {len(df)} pairs -> {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", required=True, help="Path to organized audio dir")
    parser.add_argument("--output-dir", required=True, help="Path to output metadata dir")
    args = parser.parse_args()
    
    prepare_pairs(args.audio_dir, args.output_dir)