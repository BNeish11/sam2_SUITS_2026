import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AI + remote bridge with one command.")
    parser.add_argument("--demo", default="run_video_demo.py", help="Main AI entry script")
    parser.add_argument("--forward-result", action="store_true", help="Forward JSON result to server for Unreal")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / ".env")

    # Force remote integration on for this run.
    os.environ["ENABLE_REMOTE"] = "true"

    server_url = os.getenv("SERVER_URL", "").strip()
    if not server_url or "YOUR_SERVER_IP" in server_url:
        print("[ERROR] Set SERVER_URL in .env first (example: http://192.168.1.50:3000)")
        return 1

    bridge_script = repo_root / "tools" / "remote_frame_bridge.py"
    demo_script = repo_root / args.demo

    if not bridge_script.exists():
        print(f"[ERROR] Missing bridge script: {bridge_script}")
        return 1
    if not demo_script.exists():
        print(f"[ERROR] Missing demo script: {demo_script}")
        return 1

    bridge_cmd = [sys.executable, str(bridge_script)]
    if args.forward_result:
        bridge_cmd.append("--forward-result")

    demo_cmd = [sys.executable, str(demo_script)]

    print(f"[INFO] SERVER_URL={server_url}")
    print(f"[INFO] Starting bridge: {' '.join(bridge_cmd)}")
    bridge_proc = subprocess.Popen(bridge_cmd, cwd=repo_root)

    try:
        time.sleep(1.0)
        print(f"[INFO] Starting AI demo: {' '.join(demo_cmd)}")
        demo_proc = subprocess.Popen(demo_cmd, cwd=repo_root)
        demo_return = demo_proc.wait()
        return demo_return
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
        return 130
    finally:
        if bridge_proc.poll() is None:
            bridge_proc.terminate()
            try:
                bridge_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bridge_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())