from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from typing import Iterator

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


@contextmanager
def _resolve_video_path(video: Input.Video) -> Iterator[str]:
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

    if isinstance(source, str) and trim_window == (0.0, 0.0) and crop_rect is None:
        yield source
        return

    temp_root = folder_paths.get_temp_directory()
    os.makedirs(temp_root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wemm_video_", dir=temp_root) as tmpdir:
        tmp_path = str(Path(tmpdir) / "input.mp4")
        video.save_to(
            tmp_path,
            format=Types.VideoContainer.MP4,
            codec=Types.VideoCodec.AUTO,
        )
        yield tmp_path


class WeMMVideoModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WeMMVideoModelLoader",
            display_name="WeMM Video Model Loader",
            category="video/WeMM",
            description="Loads a WeMM-Video AWQ model and processor.",
            inputs=[
                io.String.Input(
                    "model_id",
                    default="tencent/WeMM-Video-9B-AWQ",
                    tooltip="HuggingFace model ID or folder name under models/LLM.",
                ),
                io.String.Input(
                    "max_memory",
                    default="13GiB",
                    tooltip="Per-GPU budget (e.g. '13GiB'), comma-separated per GPU ('13GiB,13GiB'), or JSON dict.",
                ),
                io.Boolean.Input(
                    "run_compressed",
                    default=False,
                    advanced=True,
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
        run_compressed: bool,
    ) -> io.NodeOutput:
        model_source = _resolve_model_source(model_id)
        config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)

        max_mem_map = _parse_max_memory(max_memory)
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

        quant_cfg = getattr(config, "quantization_config", None)
        if isinstance(quant_cfg, dict) and "config_groups" in quant_cfg:
            for group in quant_cfg["config_groups"].values():
                group["targets"] = _AWQ_MLP_TARGETS

        processor = AutoProcessor.from_pretrained(
            model_source,
            trust_remote_code=True,
        )

        offload_dir = os.path.join(folder_paths.get_temp_directory(), "wemm_offload")
        os.makedirs(offload_dir, exist_ok=True)

        model = AutoModelForImageTextToText.from_pretrained(
            model_source,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            config=config,
            quantization_config=CompressedTensorsConfig(run_compressed=run_compressed),
            device_map=device_map,
            offload_folder=offload_dir,
        ).eval()

        return io.NodeOutput((model, processor))


class WeMMVideoInference(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WeMMVideoInference",
            display_name="WeMM Video Inference",
            category="video/WeMM",
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

        def _run_with_video(video_path: str | None) -> str:
            if video_path is not None:
                content.append(
                    {
                        "type": "video",
                        "video": video_path,
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

            with torch.inference_mode():
                output = loaded_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )

            answer_ids = output[0, inputs.input_ids.shape[1] :]
            return processor.tokenizer.decode(answer_ids, skip_special_tokens=True)

        if video is not None:
            with _resolve_video_path(video) as video_path:
                answer = _run_with_video(video_path)
        else:
            answer = _run_with_video(None)

        return io.NodeOutput(answer)