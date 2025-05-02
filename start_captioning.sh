source venv/bin/activate
clear
python src/captioning_app.py  --rich_captions --audio_input_device_index=1 --model moonshine_onnx_tiny

# python src/captioning_app.py --show_audio_devices