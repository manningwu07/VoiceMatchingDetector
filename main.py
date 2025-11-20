import argparse
import sys
from src.verification import VoiceVerifier

def main():
    parser = argparse.ArgumentParser(description="Human Voice Detector / Verifier")
    parser.add_argument("file1", type=str, help="Path to first audio file")
    parser.add_argument("file2", type=str, help="Path to second audio file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, mps, cuda)")

    args = parser.parse_args()

    try:
        # Initialize the engine
        verifier = VoiceVerifier(device=args.device)
        
        print(f"\nComparing:\n 1. {args.file1}\n 2. {args.file2}\n")
        
        # Run verification
        result = verifier.verify(args.file1, args.file2)

        # Output results
        decision = "SAME PERSON" if result["match"] else "DIFFERENT PEOPLE"
        color = "\033[92m" if result["match"] else "\033[91m" # Green for same, Red for diff
        reset = "\033[0m"

        print(f"Result: {color}{decision}{reset}")
        print(f"Similarity Score: {result['score']:.4f} (Threshold: 0.25)")
        print(f"Confidence: {result['confidence']*100:.2f}%")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()