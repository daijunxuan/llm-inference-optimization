import json
import subprocess
import sys
from pathlib import Path


def suite(args):
    config = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for profile in config["profiles"]:
        settings = config["common"] | profile["options"]
        name = profile["name"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name):
            raise ValueError("Profile name must be a safe lowercase filename")
        target = out / f"{name}.json"
        # A failed rerun must not leave a previous successful result under the same name.
        if target.exists():
            raise FileExistsError(f"Use a fresh output directory; {target} already exists")
        cmd = [sys.executable, "-m", "llmbench", "run", "--out", str(target)]
        for key, value in settings.items():
            cmd += ["--" + key.replace("_", "-"), str(value)]
        print(f"Running {name}", flush=True)
        with (out / f"{name}.log").open("w") as log:
            completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        manifest.append(
            {
                "profile": name,
                "exit_code": completed.returncode,
                "result": target.name if completed.returncode == 0 else None,
            }
        )
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if completed.returncode:
            print(f"FAILED: see {out / (name + '.log')}", file=sys.stderr)
            raise SystemExit(completed.returncode)
    print("Suite complete. Generate a report from the profile JSON files (exclude manifest.json).")
