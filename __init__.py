from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .nodes import WeMMVideoInference, WeMMVideoModelLoader


class WeMMVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            WeMMVideoModelLoader,
            WeMMVideoInference,
        ]


async def comfy_entrypoint() -> WeMMVideoExtension:
    return WeMMVideoExtension()