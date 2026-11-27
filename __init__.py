from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .nodes import WeMMVideoInference, WeMMVideoModelLoader
from .nodes import ShellCommandNode, RestartSignal, DataInspector, ModelDownloaderNode


class WeMMVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WeMMVideoModelLoader,
            WeMMVideoInference,
            RestartSignal,
            ShellCommandNode,
            DataInspector,
            ModelDownloaderNode,
        ]


async def comfy_entrypoint() -> WeMMVideoExtension:
    return WeMMVideoExtension()