from __future__ import annotations

import base64
import io as stdlib_io
import json
import os

from accelerate import infer_auto_device_map, init_empty_weights
import numpy as np
from PIL import Image
import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    CompressedTensorsConfig,
)

import comfy.model_management
from comfy_api.latest import Input, Types, io
import folder_paths


WEMM_MODEL = io.Custom("WEMM_VIDEO_MODEL")

_AWQ_MLP_TARGETS = [
    r"re:^model\.language_model\.layers\.\d+\.mlp\."
    r"(?:(?:experts\.\d+|shared_expert)\.)?"
    r"(?:gate|up|down)_proj$"
]

_DIRECT_DEVICE_MAPS = {"auto", "balanced", "balanced_low_0", "sequential"}


def _resolve_model_source(model_id: str) -> str:
    local_candidate = os.path.join(folder_paths.models_dir, "LLM", model_id)
    if os.path.isdir(local_candidate):
        return local_candidate
    return model_id


def _parse_max_memory(spec: str) -> dict[int | str, str] | None:
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return None

    spec = spec.strip()
    if not spec:
        return None

    if spec.startswith("{"):
        raw = json.loads(spec)
        return {(int(k) if str(k).isdigit() else k): str(v) for k, v in raw.items()}

    if "," in spec:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        return {i: parts[i] for i in range(min(gpu_count, len(parts)))}

    return {i: spec for i in range(gpu_count)}


def _video_to_data_uri(video: Input.Video) -> str:
    source = video.get_stream_source()
    trim_window = (
        video.get_active_trim_window()
        if hasattr(video, "get_active_trim_window")
        else (0.0, 0.0)
    )
    crop_rect = (
        video.get_active_crop_rect()
        if hasattr(video, "get_active_crop_rect")
        else None
    )

    if (
        isinstance(source, stdlib_io.BytesIO)
        and trim_window == (0.0, 0.0)
        and crop_rect is None
    ):
        raw_bytes = source.getvalue()
    else:
        buf = stdlib_io.BytesIO()
        video.save_to(
            buf,
            format=Types.VideoContainer.MP4,
            codec=Types.VideoCodec.AUTO,
        )
        raw_bytes = buf.getvalue()

    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


class WeMMVideoModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WeMMVideoModelLoader",
            display_name="WeMM Video Model Loader",
            category="WeMM",
            description="Loads a WeMM-Video model and processor.",
            inputs=[
                io.String.Input(
                    "model_id",
                    default="tencent/WeMM-Video-9B-AWQ",
                    tooltip="HuggingFace model ID or folder name under models/LLM.",
                ),
                io.String.Input(
                    "max_memory",
                    default="13GiB",
                    tooltip="Per-GPU budget ('13GiB', '13GiB,13GiB', JSON dict) or 'auto' / 'cuda:0'.",
                ),
            ],
            outputs=[
                WEMM_MODEL.Output("model"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_id: str,
        max_memory: str,
    ) -> io.NodeOutput:
        model_source = _resolve_model_source(model_id)
        config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)

        spec = max_memory.strip()
        if spec.lower() in _DIRECT_DEVICE_MAPS or spec.lower().startswith("cuda"):
            device_map: str | dict | None = spec
        else:
            max_mem_map = _parse_max_memory(spec)
            if max_mem_map is not None:
                with init_empty_weights():
                    empty_model = AutoModelForImageTextToText.from_config(
                        config,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                    )
                no_split = getattr(empty_model, "_no_split_modules", None)
                device_map = infer_auto_device_map(
                    empty_model,
                    max_memory=max_mem_map,
                    no_split_module_classes=no_split,
                    dtype=torch.bfloat16,
                    verbose=False,
                )
            else:
                device_map = "auto" if torch.cuda.is_available() else None

        load_kwargs: dict = {
            "trust_remote_code": True,
            "dtype": torch.bfloat16,
            "config": config,
            "device_map": device_map,
        }

        quant_cfg = getattr(config, "quantization_config", None)
        quant_method = (
            quant_cfg.get("quant_method")
            if isinstance(quant_cfg, dict)
            else getattr(quant_cfg, "quant_method", None)
        )
        if quant_method == "compressed-tensors":
            if isinstance(quant_cfg, dict) and "config_groups" in quant_cfg:
                for group in quant_cfg["config_groups"].values():
                    group["targets"] = _AWQ_MLP_TARGETS
            load_kwargs["quantization_config"] = CompressedTensorsConfig(
                run_compressed=False
            )

        if device_map == "auto" or (
            isinstance(device_map, dict) and "disk" in device_map.values()
        ):
            offload_dir = os.path.join(folder_paths.get_temp_directory(), "wemm_offload")
            os.makedirs(offload_dir, exist_ok=True)
            load_kwargs["offload_folder"] = offload_dir

        processor = AutoProcessor.from_pretrained(
            model_source,
            trust_remote_code=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_source,
            **load_kwargs,
        ).eval()

        for module in model.modules():
            if hasattr(module, "_old_forward") and not hasattr(module, "_hf_hook"):
                module.forward = module._old_forward

        return io.NodeOutput((model, processor))


class WeMMVideoInference(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WeMMVideoInference",
            display_name="WeMM Video Inference",
            category="WeMM",
            description="Runs WeMM-Video multimodal inference on video, images, and text.",
            inputs=[
                WEMM_MODEL.Input("model"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="Describe what happens on screen.",
                ),
                io.Video.Input("video", optional=True),
                io.Image.Input("image", optional=True),
                io.Float.Input("fps", default=1.0, min=0.1, max=60.0, step=0.1),
                io.Int.Input("min_frames", default=4, min=1, max=256),
                io.Int.Input("max_frames", default=16, min=1, max=256),
                io.Int.Input("total_pixels", default=262144, min=16384, max=4194304, step=1024),
                io.Int.Input("max_new_tokens", default=128, min=1, max=4096),
                io.Boolean.Input("do_sample", default=False, advanced=True),
                io.Boolean.Input("enable_thinking", default=False, advanced=True),
            ],
            outputs=[
                io.String.Output("text"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: tuple,
        prompt: str,
        fps: float,
        min_frames: int,
        max_frames: int,
        total_pixels: int,
        max_new_tokens: int,
        do_sample: bool,
        enable_thinking: bool,
        video: Input.Video | None = None,
        image: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        comfy.model_management.throw_exception_if_processing_interrupted()

        loaded_model, processor = model
        content: list[dict] = []

        if video is not None:
            content.append(
                {
                    "type": "video",
                    "video": _video_to_data_uri(video),
                    "fps": fps,
                    "min_frames": min_frames,
                    "max_frames": max_frames,
                    "total_pixels": total_pixels,
                }
            )

        if image is not None:
            frames = (image.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
            for frame in frames:
                content.append(
                    {
                        "type": "image",
                        "image": Image.fromarray(np.ascontiguousarray(frame)),
                    }
                )

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ).to(device=loaded_model.device, dtype=loaded_model.dtype)

        del content, messages

        output = None
        try:
            with torch.inference_mode():
                output = loaded_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
            answer_ids = output[0, inputs.input_ids.shape[1] :]
            answer = processor.tokenizer.decode(answer_ids, skip_special_tokens=True)
        finally:
            del inputs, output
            comfy.model_management.soft_empty_cache()

        return io.NodeOutput(answer)