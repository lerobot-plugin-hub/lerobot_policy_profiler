"""LeRobot policy profiler plugin discovery entrypoint.

LeRobot imports installed distributions whose names start with
``lerobot_policy_``. This package is the recommended public import path and
forwards to the compatibility implementation package ``lerobot_profiler``.
"""

from lerobot_profiler import *  # noqa: F403
