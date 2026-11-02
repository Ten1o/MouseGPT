# Environment installation matrix

Updated 2026-11-02. Core CPU/API and the older tool-calling Agent remain isolated because their OpenAI/LangChain dependency versions differ. Pose automatic installation is still withdrawn; validate and Use a manually configured environment.

## Current automatic installation targets

| Platform | Core | Agent | Runtime evidence |
| --- | --- | --- | --- |
| Windows x64 | Existing recipe | Existing recipe | Earlier Windows clean-install evidence retained |
| Linux x64, glibc 2.28+ | Separate locked recipe | Separate locked recipe | Ubuntu installation user-confirmed; broader Linux verification pending |
| Linux ARM64, glibc 2.28+ | Separate locked recipe | Separate locked recipe | Real-machine verification pending |
| macOS 14+ Intel | Platform-specific Numba pin | Separate locked recipe | Real-machine verification pending |
| macOS 14+ Apple Silicon | Separate locked recipe | Platform-specific PyTables/Blosc2 pins | Real-machine verification pending |

Windows ARM64, 32-bit systems, musl/Alpine Linux, glibc older than 2.28 and macOS older than 14 require manual setup. This table describes implemented installation targets, not a claim that every listed system has passed end-to-end tests. Each installed candidate still runs dependency and actual small-data checks before it can be registered and activated.

`platform-matrix.json` records ten recipe hashes, Python versions, notable dependency versions and evidence status. `validation.json` retains the earlier Windows runtime evidence. No new-platform runtime success is fabricated.

## Dependency recipes

Base direct dependencies are in core.in / agent.in. Generated lock files pin the complete closure and artifact hashes. Windows retains core.lock / agent.lock unchanged; other platforms select profile-platform.lock. Installation never resolves fresh package versions or falls back to building native packages from source.

The Intel Mac Core recipe fixes Numba at 0.61.2 because the baseline newer Numba has no matching Intel wheel. The Apple Silicon Agent recipe uses PyTables 3.10.1 and Blosc2 2.7.1 because PyTables 3.8.0/3.9.2 lack the needed native wheel. Only these platform recipes receive release 2026.11.02.1; Windows and current Linux releases are preserved.

All six newly added target recipes were resolved on Windows using uv target-platform/Python selection and binary-only packages. This checks wheel availability and metadata compatibility, not imports, native ABI, SDK behavior or real task performance on another OS. The three new uv bootstrap wheels were downloaded and hash/layout checked without executing foreign binaries.

## Isolation and validation

The fixed uv 0.8.22 bootstrap is downloaded from PyPI and verified by a pinned SHA-256. macOS/Linux tools are cached per platform, with executable permissions set after verification. Managed Python is downloaded into application storage; virtual environments use Scripts/python.exe on Windows and bin/python elsewhere, without resolving venv interpreter symlinks to the base Python.

Installation uses a new unique directory and never upgrades an existing environment in place. Dependency checks and the offline environment_probe.py small-data checks must pass before registration; Use is an explicit separate action. Failed or cancelled operations do not replace the active environment. Core validation needs 1_keypoints/demo/batch_vis.py and video_utils.py in the checkout. Installing an environment does not upload Stage code, data or model weights.

macOS defaults to ~/Library/Application Support/MouseGPT, retaining a pre-existing ~/.local/share/MouseGPT installation. Linux honors XDG_DATA_HOME; MOUSEGPT_UI_HOME can explicitly select an application directory. Existing Windows paths and environments are preserved.

The UI server still needs a host Python to start. Cross-platform environment installation does not create a standalone desktop application or support mobile devices.

See the [UI user guide](../README.md) for setup instructions and feature status.
