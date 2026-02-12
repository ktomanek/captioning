#!/usr/bin/env python3
"""
Convert HuggingFace Whisper models to ONNX format.

Creates default (float32) and int8 quantized versions.

Usage:
    python convert_hf_whisper_to_onnx.py openai/whisper-tiny ./onnx_models/whisper-tiny

Output structure:
    <output_folder>/
        default/           # Float32 ONNX models
        default_int8/      # Int8 quantized ONNX models
"""

from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from onnxruntime.quantization import QuantType, quantize_dynamic
import glob
import os
import shutil
import argparse


def convert_to_onnx(model_path, output_folder):
    """
    Convert HuggingFace Whisper model to ONNX format.

    Args:
        model_path: HuggingFace model ID (e.g., "openai/whisper-tiny") or local path
        output_folder: Directory to save ONNX models
    """
    print(f"\n>>> Converting {model_path} to ONNX...")
    ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        export=True,
    )
    ort_model.save_pretrained(output_folder)
    print(f"✓ Default ONNX models saved to {output_folder}")


def copy_config_files(source_dir, target_dir):
    """Copy JSON config files from source to target directory."""
    for json_file in glob.glob(os.path.join(source_dir, "*.json")):
        shutil.copy2(json_file, target_dir)


def quantize_onnx_int8(input_dir, output_dir):
    """
    Quantize ONNX models to int8.

    Args:
        input_dir: Directory containing ONNX models to quantize
        output_dir: Directory to save quantized models
    """
    print(f"\n>>> Quantizing models from {input_dir} to int8...")

    onnx_model_paths = glob.glob(os.path.join(input_dir, "*.onnx"))

    if not onnx_model_paths:
        print(f"Warning: No ONNX models found in {input_dir}")
        return

    for model_path in onnx_model_paths:
        base_name = os.path.basename(model_path)
        output_path = os.path.join(output_dir, base_name)

        print(f"  - Quantizing {base_name}...")
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            # Only quantize MatMul operations (quantizing Conv causes inference failures)
            op_types_to_quantize=['MatMul'],
            weight_type=QuantType.QInt8,
            extra_options={'DefaultTensorType': 1}  # 1 = FLOAT
        )

    # Copy config files to quantized directory
    copy_config_files(input_dir, output_dir)
    print(f"✓ Quantized models saved to {output_dir}")


def main(model_path, output_folder):
    """
    Convert HuggingFace Whisper model to ONNX with default and int8 quantized versions.

    Args:
        model_path: HuggingFace model ID or local path (e.g., "openai/whisper-tiny")
        output_folder: Base output directory
    """
    # Create output directories
    default_folder = os.path.join(output_folder, 'default')
    int8_folder = os.path.join(output_folder, 'default_int8')

    os.makedirs(default_folder, exist_ok=True)
    os.makedirs(int8_folder, exist_ok=True)

    # Convert to ONNX
    convert_to_onnx(model_path, default_folder)

    # Quantize to int8
    quantize_onnx_int8(default_folder, int8_folder)

    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"{'='*60}")
    print(f"Default (float32) models: {default_folder}")
    print(f"Int8 quantized models:    {int8_folder}")
    print(f"\nTo use with captioning app:")
    print(f"  python src/captioning_lib/captioning_app.py -m whisperonnx --model_path {default_folder} -rc -i 0")
    print(f"  python src/captioning_lib/captioning_app.py -m whisperonnx --model_path {int8_folder} -rc -i 0")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert HuggingFace Whisper model to ONNX format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='HuggingFace model ID (e.g., openai/whisper-tiny) or local path'
    )
    parser.add_argument(
        'output_folder',
        type=str,
        help='Output folder for ONNX models'
    )

    args = parser.parse_args()
    main(args.model_path, args.output_folder)
