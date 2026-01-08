"""
Download Silero VAD ONNX model from GitHub and store it locally.
"""
import urllib.request
import ssl
import hashlib
import subprocess
import sys
from pathlib import Path


# Model configurations
MODELS = {
    'silero_vad.onnx': {
        'url': 'https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx',
        'sha256': None,  # Optional: add checksum for verification
        'description': 'Standard Silero VAD ONNX model (opset 16, ~2.2MB)'
    },
    'silero_vad_half.onnx': {
        'url': 'https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad_half.onnx',
        'sha256': None,
        'description': 'Half-precision Silero VAD ONNX model (~1.2MB)'
    }
}

DEFAULT_MODEL_DIR = Path(__file__).parent / 'models' / 'silero_vad'


def download_model(model_name='silero_vad.onnx', output_dir=None, force=False):
    """
    Download Silero VAD ONNX model.

    Args:
        model_name: Name of the model to download (see MODELS dict)
        output_dir: Directory to save the model (default: ./models/silero_vad/)
        force: If True, re-download even if file exists

    Returns:
        Path to downloaded model file
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    # Set output directory
    if output_dir is None:
        output_dir = DEFAULT_MODEL_DIR
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / model_name

    # Check if already exists
    if output_path.exists() and not force:
        print(f"✓ Model already exists: {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"  Use force=True to re-download")
        return output_path

    # Download
    model_info = MODELS[model_name]
    print(f"Downloading {model_name}...")
    print(f"  Description: {model_info['description']}")
    print(f"  URL: {model_info['url']}")
    print(f"  Destination: {output_path}")

    try:
        # Try urllib first
        try:
            # Create SSL context that doesn't verify certificates (for GitHub raw content)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # Download with progress
            def reporthook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(downloaded * 100 / total_size, 100)
                    mb_downloaded = downloaded / 1024 / 1024
                    mb_total = total_size / 1024 / 1024
                    print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.2f}/{mb_total:.2f} MB)", end='')

            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(model_info['url'], output_path, reporthook=reporthook)
            print()  # New line after progress

        except Exception as urllib_error:
            print(f"\n  urllib failed: {urllib_error}")
            print(f"  Trying curl as fallback...")

            # Fallback to curl
            result = subprocess.run(
                ['curl', '-L', '-o', str(output_path), model_info['url']],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"curl failed: {result.stderr}")

            print(f"  ✓ Downloaded successfully using curl")

        # Verify checksum if provided
        if model_info['sha256']:
            print(f"  Verifying checksum...")
            sha256_hash = hashlib.sha256()
            with open(output_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

            actual_hash = sha256_hash.hexdigest()
            if actual_hash != model_info['sha256']:
                output_path.unlink()  # Delete corrupted file
                raise ValueError(f"Checksum mismatch! Expected {model_info['sha256']}, got {actual_hash}")
            print(f"  ✓ Checksum verified")

        print(f"✓ Download complete!")
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        return output_path

    except Exception as e:
        if output_path.exists():
            output_path.unlink()  # Clean up partial download
        raise RuntimeError(f"Failed to download model: {e}")


def download_all_models(output_dir=None, force=False):
    """Download all available Silero VAD models."""
    print("=" * 70)
    print("Downloading all Silero VAD ONNX models")
    print("=" * 70)

    downloaded_paths = {}
    for model_name in MODELS.keys():
        print()
        try:
            path = download_model(model_name, output_dir, force)
            downloaded_paths[model_name] = path
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")

    print()
    print("=" * 70)
    print("Summary:")
    for model_name, path in downloaded_paths.items():
        print(f"  ✓ {model_name}: {path}")
    print("=" * 70)

    return downloaded_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Silero VAD ONNX models")
    parser.add_argument(
        '--model',
        type=str,
        default='silero_vad.onnx',
        choices=list(MODELS.keys()) + ['all'],
        help='Model to download (default: silero_vad.onnx)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=f'Output directory (default: {DEFAULT_MODEL_DIR})'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if file exists'
    )

    args = parser.parse_args()

    if args.model == 'all':
        download_all_models(args.output_dir, args.force)
    else:
        download_model(args.model, args.output_dir, args.force)
