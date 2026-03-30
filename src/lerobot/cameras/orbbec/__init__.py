from .configuration_orbbec import OrbbecColorCameraConfig, OrbbecDepthCameraConfig

try:
    from .camera_orbbec import (
        OrbbecColorCamera,
        OrbbecDepthCamera,
        SharedOrbbecColorCamera,
        SharedOrbbecDepthCamera,
        SharedOrbbecManager,
        find_orbbec_cameras,
    )
except Exception:
    # Allow importing Orbbec config classes even when pyorbbecsdk is not installed.
    pass
