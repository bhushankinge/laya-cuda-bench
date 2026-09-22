"""Write manifest.json: pinned source shas, HF revisions, SHA256 of every weight file. Stdlib only.

    python scripts/make_manifest.py
"""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOS = {"laya": "NandhaKishorM/laya", "laya-mlx": "mizorewww/laya-mlx", "receptron-laya": "receptron/laya"}
HF = {"laya": "convaiinnovations/laya", "laya-multilingual": "convaiinnovations/laya-multilingual",
      "laya-typed-decisions": "convaiinnovations/laya-typed-decisions"}
REQUESTED_HF_REVISION = {"laya": "c5d78730f3493e4fe16d61507ef4b78eef7318cf"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = {"sources": {}, "checkpoints": {}}
    for name, repo in REPOS.items():
        d = ROOT / "third_party" / name
        manifest["sources"][name] = {"github": repo,
                                     "sha": subprocess.check_output(["git", "-C", str(d), "rev-parse", "HEAD"], text=True).strip()}
    for name, repo in HF.items():
        d = ROOT / "models" / name
        files = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix in (".safetensors", ".json") and "onnx" not in p.parts)
        manifest["checkpoints"][name] = {
            "hf_repo": repo,
            "requested_revision": REQUESTED_HF_REVISION.get(name, "main"),
            "files": {str(p.relative_to(d)): {"bytes": p.stat().st_size, **({"sha256": sha256(p)} if p.suffix == ".safetensors" else {})}
                      for p in files},
            "rl_agent_config": json.loads((d / "rl_agent_config.json").read_text()),
        }
    out = ROOT / "manifest.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    print("wrote", out)
    for n, c in manifest["checkpoints"].items():
        print(n, {k: v.get("sha256", "")[:12] for k, v in c["files"].items() if "sha256" in v})


if __name__ == "__main__":
    main()
