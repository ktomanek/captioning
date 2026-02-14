#!/usr/bin/env python3
"""
Convert HuggingFace Whisper models to CTranslate2 format for faster-whisper.

Creates int8 quantized version by default for optimal performance.

Usage:
    python convert_hf_whisper_to_ct2.py hf_repo/model_name /tmp/model_name

"""

import argparse
import os
from ctranslate2.converters import TransformersConverter


def convert_to_ct2(model_path, output_folder, quantization="int8"):
    """
    Convert HuggingFace Whisper model to CTranslate2 format.

    Args:
        model_path: HuggingFace model ID (e.g., "openai/whisper-tiny") or local path
        output_folder: Directory to save CTranslate2 model
        quantization: Quantization type (default: "int8", options: "int8", "int8_float16", "float16", "float32")
    """
    print(f"\n>>> Converting {model_path} to CTranslate2 format...")
    print(f"    Quantization: {quantization}")

    os.makedirs(output_folder, exist_ok=True)

    converter = TransformersConverter(model_path)
    converter.convert(
        output_dir=output_folder,
        quantization=quantization,
        force=True
    )

    print(f"✓ CTranslate2 model saved to {output_folder}")


def main(model_path, output_folder, quantization="int8"):
    """
    Convert HuggingFace Whisper model to CTranslate2 format for faster-whisper.

    Args:
        model_path: HuggingFace model ID or local path (e.g., "openai/whisper-tiny")
        output_folder: Output directory for CTranslate2 model
        quantization: Quantization type (default: "int8")
    """
    convert_to_ct2(model_path, output_folder, quantization)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert HuggingFace Whisper model to CTranslate2 format for faster-whisper',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='HuggingFace model ID (e.g., hf_repo/model_name) or local path'
    )
    parser.add_argument(
        'output_folder',
        type=str,
        help='Output folder for CTranslate2 model'
    )
    parser.add_argument(
        '--quantization',
        type=str,
        default='int8',
        choices=['int8', 'int8_float16', 'float16', 'float32'],
        help='Quantization type (default: int8)'
    )

    args = parser.parse_args()
    main(args.model_path, args.output_folder, args.quantization)
