import gradio as gr
import torch
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download, snapshot_download

gen_kwargs = {
    "task": "transcribe",
    "language": "fr",
    "vad_filter": True,
}

DEFAULT_MODEL_NAME = "bofenghuang/whisper-large-v3-french"

cached_models = {}

device = "cuda" if torch.cuda.is_available() else "cpu"

device_compute_type = {"cuda" : "float16", "cpu" : "int8"}

def maybe_load_cached_pipeline(model_name):
    model = cached_models.get(model_name)
    if model is None:
        downloaded_model_path = snapshot_download(repo_id=model_name, allow_patterns="ctranslate2/*")
        downloaded_model_path = f"{downloaded_model_path}/ctranslate2"
        model = WhisperModel(downloaded_model_path, device=device, compute_type=device_compute_type[device])

        cached_models[model_name] = model
    return model


def infer(filename):
  model = maybe_load_cached_pipeline(DEFAULT_MODEL_NAME)
  model_outputs, _ = model.transcribe(filename, without_timestamps=True, **gen_kwargs)
  transcription = " ".join([segment.text for segment in model_outputs])
  return transcription


description = '''Whisper Demo with a fine-tuned checkpoint by bofenghuang'''

iface = gr.Interface(fn=infer,
                     inputs=[
                         gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or upload file")
                         ],
                     outputs=gr.Textbox(label="Transcription"),
                     description=description
                     )
iface.launch()
