from typing import NamedTuple
import argparse
import os


class Args(NamedTuple):
    host: str
    port: int
    reload: bool
    mode: str
    max_queue_size: int
    timeout: float
    safety_checker: bool
    taesd: bool
    ssl_certfile: str
    ssl_keyfile: str
    debug: bool
    acceleration: str
    engine_dir: str
    controlnet_config: str
    api_only: bool
    log_level: str
    quiet: bool

    def pretty_print(self):
        print("\n")
        for field, value in self._asdict().items():
            print(f"{field}: {value}")
        print("\n")


MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 0))
TIMEOUT = float(os.environ.get("TIMEOUT", 0))
SAFETY_CHECKER = os.environ.get("SAFETY_CHECKER", None) == "True"
ENGINE_DIR = os.environ.get("ENGINE_DIR", "engines")
ACCELERATION = os.environ.get("ACCELERATION", "xformers")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
QUIET = os.environ.get("QUIET", "False").lower() in ("true", "1", "yes", "on")

default_host = os.getenv("HOST", "0.0.0.0")
default_port = int(os.getenv("PORT", "7860"))
default_mode = os.getenv("MODE", "default")

parser = argparse.ArgumentParser(description="Run the app")
parser.add_argument("--host", type=str, default=default_host, help="Host address")
parser.add_argument("--port", type=int, default=default_port, help="Port number")
parser.add_argument("--reload", action="store_true", help="Reload code on change")
parser.add_argument(
    "--mode", type=str, default=default_mode, help="App Inferece Mode: txt2img, img2img"
)
parser.add_argument(
    "--max-queue-size",
    dest="max_queue_size",
    type=int,
    default=MAX_QUEUE_SIZE,
    help="Max Queue Size",
)
parser.add_argument("--timeout", type=float, default=TIMEOUT, help="Timeout")
parser.add_argument(
    "--safety-checker",
    dest="safety_checker",
    action="store_true",
    default=SAFETY_CHECKER,
    help="Safety Checker",
)
parser.add_argument(
    "--taesd",
    dest="taesd",
    action="store_true",
    help="Use Tiny Autoencoder",
)
parser.add_argument(
    "--no-taesd",
    dest="taesd",
    action="store_false",
    help="Use Tiny Autoencoder",
)
parser.add_argument(
    "--ssl-certfile",
    dest="ssl_certfile",
    type=str,
    default=None,
    help="SSL certfile",
)
parser.add_argument(
    "--ssl-keyfile",
    dest="ssl_keyfile",
    type=str,
    default=None,
    help="SSL keyfile",
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Debug",
)
parser.add_argument(
    "--acceleration",
    type=str,
    default=ACCELERATION,
    choices=["none", "xformers", "sfast", "tensorrt"],
    help="Acceleration",
)
parser.add_argument(
    "--engine-dir",
    dest="engine_dir",
    type=str,
    default=ENGINE_DIR,
    help="Engine Dir",
)
parser.add_argument(
    "--controlnet-config",
    dest="controlnet_config",
    type=str,
    default=None,
    help="Path to ControlNet YAML configuration file (optional)",
)
parser.add_argument(
    "--api-only",
    dest="api_only",
    action="store_true",
    default=False,
    help="Run API only without serving frontend static files (useful for development with separate Vite dev server)",
)
parser.add_argument(
    "--log-level",
    dest="log_level",
    type=str,
    default=LOG_LEVEL,
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
parser.add_argument(
    "--quiet",
    dest="quiet",
    action="store_true",
    default=QUIET,
    help="Suppress uvicorn INFO messages (server access logs, etc.)",
)
config = Args(**vars(parser.parse_args()))
config.pretty_print()
