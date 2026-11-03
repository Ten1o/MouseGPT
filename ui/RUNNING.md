# Running MouseGPT UI

The maintained user guide is [README.md](README.md), including Windows/macOS/Linux startup, environment and API configuration, stage inputs/outputs, remote access and troubleshooting.

Installation recipes and evidence are described in [environments/README.md](environments/README.md).

The UI service requires Python 3.10+; it does not require Node.js/npm. On Windows run ui/start-ui.cmd. On macOS/Linux run python3 ui/server.py --port 8878. Keep the terminal open and visit http://127.0.0.1:8878/ui/.
