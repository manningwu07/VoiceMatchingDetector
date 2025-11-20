import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from torch.nn import CosineSimilarity
import os

class VoiceVerifier:
    def __init__(self, device: str = "cpu"):
        print(f"Loading ECAPA-TDNN model on {device}...")
        # Downloads the pretrained model from HuggingFace automatically
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        self.cosine_sim = CosineSimilarity(dim=-1, eps=1e-6)
        self.device = device
        
        # Calibrated threshold for ECAPA-TDNN on VoxCeleb is typically around 0.25
        # We use this to determine the decision boundary.
        self.threshold = 0.25

    def load_and_preprocess(self, path: str) -> torch.Tensor:
        """
        Loads audio and ensures it fits the model's expected format.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        signal, fs = torchaudio.load(path)

        # The model expects 16kHz audio. Resample if necessary.
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            signal = resampler(signal)

        # If stereo, convert to mono
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        return signal.to(self.device)

    def _to_probability(self, score: float) -> float:
        """
        Heuristic to map cosine similarity (-1 to 1) to a confidence probability (0 to 1).
        We use a sigmoid function centered around the threshold.
        """
        # Scaling factor determines how "sharp" the transition is. 
        # 10 is an empirical choice for this specific embedding space.
        scale = 10.0 
        shift = self.threshold
        
        probability = 1 / (1 + torch.exp(-torch.tensor(scale * (score - shift))))
        return probability.item()

    def verify(self, path_a: str, path_b: str):
        """
        Compares two audio files and returns decision logic.
        """
        # 1. Preprocess
        wav_a = self.load_and_preprocess(path_a)
        wav_b = self.load_and_preprocess(path_b)

        # 2. Generate Embeddings
        # The classifier expects a batch dimension
        emb_a = self.classifier.encode_batch(wav_a)
        emb_b = self.classifier.encode_batch(wav_b)

        # 3. Calculate Cosine Similarity
        # similarity score is usually between -1 and 1
        score = self.cosine_sim(emb_a, emb_b).item()

        # 4. Determine Decision and Confidence
        is_same_person = score > self.threshold
        
        # Calculate confidence/probability
        # If score is exactly threshold, probability is 50%
        probability = self._to_probability(score)
        
        # If we predict "Different", invert the probability to represent 
        # "Confidence that they are different"
        display_confidence = probability if is_same_person else (1 - probability)

        return {
            "match": is_same_person,
            "score": score,
            "probability": probability, # Raw probability of being "Same"
            "confidence": display_confidence # Confidence in the specific decision
        }