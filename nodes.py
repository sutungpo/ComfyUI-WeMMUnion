from __future__ import annotations

import base64
import io as stdlib_io
import json
import gc
import os
import re
import torch
import shutil
import folder_paths
import subprocess
import sys
import time
import itertools
import numpy as np
from PIL import Image
import urllib.parse

from accelerate import infer_auto_device_map, init_empty_weights
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    CompressedTensorsConfig,
)

import comfy.model_management
from comfy_api.latest import Input, Types, io
import logging
logger = logging.getLogger(__name__)

import threading
from accelerate.hooks import remove_hook_from_module

WEMM_MODEL = io.Custom("WEMM_VIDEO_MODEL")

_AWQ_MLP_TARGETS = [
    r"re:^model\.language_model\.layers\.\d+\.mlp\."
    r"(?:(?:experts\.\d+|shared_expert)\.)?"
    r"(?:gate|up|down)_proj$"
]

_DIRECT_DEVICE_MAPS = {"auto", "balanced", "balanced_low_0", "sequential"}

_SLOT_LOCK = threading.Lock()
_ACTIVE_SLOT: dict[str, object] = {
    "spec": None,
    "model": None,
    "processor": None,
}


def _resolve_model_source(model_id: str, local_path: str) -> tuple[str, bool]:
    raw_local = local_path.strip()
    if raw_local:
        expanded = os.path.expanduser(raw_local)
        if os.path.isdir(expanded):
            return os.path.abspath(expanded), True
        llm_relative = os.path.join(folder_paths.models_dir, "LLM", expanded)
        if os.path.isdir(llm_relative):
            return os.path.abspath(llm_relative), True
        raise FileNotFoundError(f"Local model directory not found: {expanded}")

    raw_id = model_id.strip()
    expanded_id = os.path.expanduser(raw_id)
    if os.path.isdir(expanded_id):
        return os.path.abspath(expanded_id), True

    llm_candidate = os.path.join(folder_paths.models_dir, "LLM", raw_id)
    if os.path.isdir(llm_candidate):
        return os.path.abspath(llm_candidate), True

    return raw_id, False


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


def _unload_active_model() -> None:
    old_model = _ACTIVE_SLOT["model"]
    _ACTIVE_SLOT["spec"] = None
    _ACTIVE_SLOT["model"] = None
    _ACTIVE_SLOT["processor"] = None

    if old_model is not None:
        remove_hook_from_module(old_model, recurse=True)
        del old_model

    gc.collect()
    comfy.model_management.unload_all_models()
    if torch.cuda.is_available():
        for dev in range(torch.cuda.device_count()):
            torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        usage = ", ".join(
            f"cuda:{i}={torch.cuda.memory_allocated(i) / (1024**3):.2f}GiB"
            for i in range(torch.cuda.device_count())
        )
        logger.info(f"[WeMM] VRAM after unload: {usage}")
    else:
        comfy.model_management.soft_empty_cache()


def _ensure_loaded(spec: tuple[str, bool, str]) -> tuple[object, object]:
    with _SLOT_LOCK:
        if (
            _ACTIVE_SLOT["spec"] == spec
            and _ACTIVE_SLOT["model"] is not None
            and _ACTIVE_SLOT["processor"] is not None
        ):
            return _ACTIVE_SLOT["model"], _ACTIVE_SLOT["processor"]

        _unload_active_model()

        model_source, local_only, max_memory_spec = spec
        config = AutoConfig.from_pretrained(
            model_source,
            trust_remote_code=True,
            local_files_only=local_only,
        )

        if (
            max_memory_spec.lower() in _DIRECT_DEVICE_MAPS
            or max_memory_spec.lower().startswith("cuda")
        ):
            device_map: str | dict | None = max_memory_spec
        else:
            max_mem_map = _parse_max_memory(max_memory_spec)
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
            "local_files_only": local_only,
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
            local_files_only=local_only,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_source,
            **load_kwargs,
        ).eval()

        for module in model.modules():
            if hasattr(module, "_old_forward") and not hasattr(module, "_hf_hook"):
                module.forward = module._old_forward

        _ACTIVE_SLOT["spec"] = spec
        _ACTIVE_SLOT["model"] = model
        _ACTIVE_SLOT["processor"] = processor
        return model, processor


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
            description="Loads a WeMM-Video model from HuggingFace or a local directory.",
            inputs=[
                io.String.Input(
                    "model_id",
                    default="tencent/WeMM-Video-9B-AWQ",
                    tooltip="HuggingFace model ID or folder name under models/LLM (used when local_path is empty).",
                ),
                io.String.Input(
                    "local_path",
                    default="",
                    tooltip="Optional local directory path of an already downloaded model (skips downloading).",
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
        local_path: str,
        max_memory: str,
    ) -> io.NodeOutput:
        model_source, local_only = _resolve_model_source(model_id, local_path)
        spec = (model_source, local_only, max_memory.strip())
        _ensure_loaded(spec)
        return io.NodeOutput(spec)


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
        model: tuple[str, bool, str],
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

        loaded_model, processor = _ensure_loaded(model)
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

########################################################
# ShellCommandNode
########################################################
class ShellCommandNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ShellCommandRunner",
            display_name="Shell Command Runner",
            category="utils",
            description="Execute shell commands on the host system and log output.",
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                io.String.Input(
                    "command",
                    default="",
                    multiline=True,
                    tooltip="Shell command(s) to execute.",
                ),
                io.String.Input(
                    "cwd",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip="Working directory (defaults to custom_nodes when empty).",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> int:
        return time.monotonic_ns()

    @classmethod
    def execute(cls, command: str, cwd: str = "") -> io.NodeOutput:
        cmd = (command or "").strip()
        if not cmd:
            raise ValueError("Command cannot be empty.")

        work_dir = (cwd or "").strip() or folder_paths.get_folder_paths("custom_nodes")[0]
        if not os.path.isdir(work_dir):
            raise FileNotFoundError(f"Working directory does not exist: {work_dir}")

        env = os.environ.copy()
        py_bin = os.path.dirname(sys.executable)
        env["PATH"] = f"{py_bin}{os.pathsep}{env.get('PATH', '')}"

        logger.info("Executing command in %s: %s", work_dir, cmd)
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
        )

        if result.stdout:
            logger.info("[stdout]\n%s", result.stdout.rstrip())
        if result.stderr:
            logger.warning("[stderr]\n%s", result.stderr.rstrip())
        logger.info("Command exited with code %d", result.returncode)

        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        return io.NodeOutput()

# =====================================================================
# RestartSignal
# =====================================================================
class RestartSignal(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RestartSignal",
            category="utils",
            is_output_node=True,
            inputs=[],
            outputs=[],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        flag = "/tmp/comfyui_restart.request"
        with open(flag, "w") as f:
            f.write("restart")

        logger.warning("[RestartSignal] Restart requested.")
        return io.NodeOutput()

# =====================================================================
# DataInspector log
# =====================================================================
"""Data Inspector: a pass-through node that logs details of whatever flows through it."""
MAX_VALUE_ELEMS = 32  # larger values are not logged; also caps listed container items
MAX_DEPTH = 2         # container levels expanded below the input

_DEVICE_ATTRS = ("device", "load_device", "offload_device", "current_device")
_WRAPPED_ATTRS = ("model", "patcher")  # one wrapper level, e.g. ModelPatcher.model, CLIP.patcher
_SCALARS = (bool, int, float, complex, type(None))


def _get(obj, name):
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _device_info(obj):
    """Best-effort device details for objects that are not tensors or plain containers."""
    info = []
    targets = [("", obj)] + [(f"{n}.", _get(obj, n)) for n in _WRAPPED_ATTRS]
    for prefix, target in targets:
        if target is None:
            continue
        for attr in _DEVICE_ATTRS:
            value = _get(target, attr)
            if value is not None and not callable(value):
                info.append(f"{prefix}{attr}={value}")
        if isinstance(target, torch.nn.Module):
            devices, params, size = set(), 0, 0
            for p in target.parameters():
                devices.add(str(p.device))
                params += p.numel()
                size += p.numel() * p.element_size()
            info.append(f"{prefix}param_devices={sorted(devices)} params={params} param_bytes={size}")
    return info


def _inspect(obj):
    """Return (one-line summary, [(child name, child), ...])."""
    parts = [f"type={type(obj).__name__}"]
    children = []
    count = value = None  # count: number of elements; value: zero-arg callable

    if isinstance(obj, torch.Tensor):
        count = obj.numel()
        parts += [f"shape={tuple(obj.shape)}", f"dtype={obj.dtype}", f"device={obj.device}",
                  f"numel={count}", f"bytes={count * obj.element_size()}"]
        value = lambda: obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        count = obj.size
        parts += [f"shape={obj.shape}", f"dtype={obj.dtype}", f"numel={count}", f"bytes={obj.nbytes}"]
        value = obj.tolist
    elif isinstance(obj, (str, bytes)):
        count = len(obj)
        parts.append(f"len={count}")
        value = lambda: obj
    elif isinstance(obj, _SCALARS):
        count, value = 1, lambda: obj
    elif isinstance(obj, (dict, list, tuple)):
        parts.append(f"len={len(obj)}")
        if len(obj) > MAX_VALUE_ELEMS:
            parts.append(f"showing_first={MAX_VALUE_ELEMS}")
        items = obj.items() if isinstance(obj, dict) else enumerate(obj)
        children = [(f"[{k!r}]", v) for k, v in itertools.islice(items, MAX_VALUE_ELEMS)]
    else:
        parts.append(f"sizeof={sys.getsizeof(obj)}")
        parts += _device_info(obj)

    if count is not None:
        parts.append(f"value={value()!r}" if count <= MAX_VALUE_ELEMS
                     else f"value=<omitted: {count} elements>")
    return " ".join(parts), children


def _describe(obj, name, depth=0):
    pad = "  " * depth
    try:
        summary, children = _inspect(obj)
    except Exception as exc:  # logging must never break the graph
        summary, children = f"type={type(obj).__name__} <inspect failed: {exc!r}>", []
    logger.info("[DataInspector] %s%s: %s", pad, name, summary)
    if children and depth >= MAX_DEPTH:
        logger.info("[DataInspector] %s  ...<max depth %d reached>", pad, MAX_DEPTH)
        return
    for child_name, child in children:
        _describe(child, child_name, depth + 1)


class DataInspector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DataInspector",
            category="utils",
            is_output_node=True,
            inputs=[io.AnyType.Input("data")],
            outputs=[io.AnyType.Output(display_name="data")],
        )

    @classmethod
    def execute(cls, data) -> io.NodeOutput:
        _describe(data, "data")
        return io.NodeOutput(data)


# =====================================================================
# Model Downloader
# =====================================================================
class ModelDownloaderNode(io.ComfyNode):
    """
    High-speed model downloader for ComfyUI leveraging aria2c.
    Supports existing categories or creating new first-level directories in ComfyUI/models/.
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        # Query ComfyUI's registry and add [models_root] to allow 1st-level folder creation
        registered_categories = sorted(list(folder_paths.folder_names_and_paths.keys()))
        available_categories = ["[models_root]"] + registered_categories
        default_category = "checkpoints" if "checkpoints" in registered_categories else "[models_root]"

        return io.Schema(
            node_id="ModelDownloaderNode",
            category="utils",
            is_output_node=True,
            inputs=[
                io.String.Input("url", default="", multiline=False),
                io.Combo.Input("target_category", options=available_categories, default=default_category),
                io.Boolean.Input("overwrite", default=False),
                io.String.Input("subfolder", default="", multiline=False, optional=True),
                io.String.Input("filename", default="", multiline=False, optional=True),
                io.String.Input("auth_token", default="", multiline=False, optional=True),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, url: str, target_category: str, overwrite: bool, subfolder: str = "", filename: str = "", auth_token: str = "") -> io.NodeOutput:
        url = url.strip()
        if not url:
            logger.info("[ModelDownloader] Empty URL provided. Skipping.")
            return io.NodeOutput()

        # 1. Platform Binary Verification
        if not shutil.which("aria2c"):
            raise RuntimeError(
                "[ModelDownloader] 'aria2c' binary not found in system PATH. "
                "Please install it in your cloud container (e.g., 'apt-get update && apt-get install -y aria2')."
            )

        # 2. Strict ComfyUI Path Resolution
        if target_category == "[models_root]":
            # Target ComfyUI/models/ directly
            base_dir = os.path.abspath(folder_paths.models_dir)
        else:
            base_dirs = folder_paths.get_folder_paths(target_category)
            if not base_dirs:
                raise ValueError(f"[ModelDownloader] No paths registered for category '{target_category}'")
            base_dir = os.path.abspath(base_dirs[0])

        clean_subfolder = os.path.normpath(subfolder.strip()).lstrip("/\\")

        if clean_subfolder in ("", "."):
            target_dir = base_dir
        else:
            target_dir = os.path.abspath(os.path.join(base_dir, clean_subfolder))

        # Enforce canonical path containment (prevent directory traversal)
        if os.path.commonpath([base_dir, target_dir]) != base_dir:
            raise ValueError(f"[ModelDownloader] Security violation: Destination '{target_dir}' escapes '{base_dir}'")

        # Automatically create the first-level or nested subfolder
        os.makedirs(target_dir, exist_ok=True)

        # If a new first-level directory is created under models/, register it with ComfyUI
        if target_category == "[models_root]" and clean_subfolder and hasattr(folder_paths, "add_model_folder_path"):
            folder_paths.add_model_folder_path(clean_subfolder, target_dir)

        # 3. Filename & Source URL Normalization
        if "huggingface.co" in url:
            url = url.replace("/blob/", "/resolve/")

        # Incur filename: User Override -> URL Path Basename -> Fallback to Content-Disposition
        target_filename = filename.strip()
        if not target_filename:
            parsed_path = urllib.parse.urlsplit(url).path.rstrip("/")
            extracted_name = os.path.basename(parsed_path)
            if extracted_name and "." in extracted_name:
                target_filename = urllib.parse.unquote(extracted_name)

        # Skip download if the final target file already exists
        if target_filename:
            destination_file = os.path.join(target_dir, target_filename)
            if os.path.exists(destination_file) and not overwrite:
                logger.info(f"[ModelDownloader] File already exists, skipping: {destination_file}")
                if hasattr(folder_paths, "filename_list_cache"):
                    folder_paths.filename_list_cache.clear()
                return io.NodeOutput()

        # 4. Assemble aria2c Command
        cmd = [
            "aria2c",
            "--console-log-level=warn",
            "--summary-interval=1",
            "--content-disposition=true",
            "-x", "16",
            "-s", "16",
            "-k", "1M",
            "-c",
            f"--dir={target_dir}",
        ]

        if target_filename:
            cmd.append(f"--out={target_filename}")

        if overwrite:
            cmd.append("--allow-overwrite=true")
        else:
            cmd.append("--auto-file-renaming=false")

        token = auth_token.strip()
        if token:
            cmd.append(f"--header=Authorization: Bearer {token}")

        cmd.append(url)

        # 5. Process Execution & Progress Reporting via Native ComfyUI Contract
        logger.info(f"[ModelDownloader] Initiating transfer for: {url}")
        pbar = comfy.utils.ProgressBar(100)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        progress_regex = re.compile(r'\((\s*\d+)%\)')

        for line in process.stdout:
            match = progress_regex.search(line)
            if match:
                percent = int(match.group(1))
                pbar.update_absolute(percent, 100)

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"[ModelDownloader] aria2c transfer failed with exit code {process.returncode}")

        pbar.update_absolute(100, 100)

        # 6. Invalidate ComfyUI Cache
        if hasattr(folder_paths, "filename_list_cache"):
            folder_paths.filename_list_cache.clear()

        logger.info(f"[ModelDownloader] Download completed successfully to {target_dir}")
        return io.NodeOutput()