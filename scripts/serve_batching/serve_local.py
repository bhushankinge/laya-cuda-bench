"""Run laya.serve (whichever laya is on PYTHONPATH) with "english" pointed at a local checkpoint dir.

    LAYA_MODEL_DIR=<checkpoint dir> PYTHONPATH=<laya checkout> python serve_local.py <port>
Batching knobs are read by laya.serve itself from LAYA_BATCH_WINDOW_MS / LAYA_BATCH_MAX (PR #434).
"""
import os
import sys

import uvicorn
from laya.router import Router
from laya.serve import create_app

router = Router(models={"english": os.environ["LAYA_MODEL_DIR"]}, device="cuda")
router.preload(["english"])
uvicorn.run(create_app(router=router), host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
