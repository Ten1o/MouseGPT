# Environment installation

Core CPU/API and the tool-calling Agent use separate, reproducible environments. The Agent recipe shares the public `4_NLUI/requirements.txt` contract and uses Python 3.12, Gradio 6 and LangChain Classic. Pose estimation still requires a manually prepared environment with the appropriate GPU libraries, models and calibration tools.

## Installation targets

| Platform | Core | Agent |
| --- | --- | --- |
| Windows x64 | Hash-locked recipe | Hash-locked Python 3.12 recipe |
| Linux x64, glibc 2.28+ | Hash-locked recipe | Hash-locked Python 3.12 recipe |
| Linux ARM64, glibc 2.28+ | Hash-locked recipe | Hash-locked Python 3.12 recipe |
| macOS 14+ Intel | Platform-specific Numba pin | Hash-locked Python 3.12 recipe |
| macOS 14+ Apple Silicon | Hash-locked recipe | Hash-locked Python 3.12 recipe |

Windows ARM64, 32-bit systems, musl/Alpine Linux, glibc older than 2.28 and macOS older than 14 require manual setup. Each environment must pass its local checks before it can be selected. Resolving a platform recipe establishes package and wheel availability; it does not establish execution on that platform or compatibility with every model service.

The current Apple Silicon Agent recipe has passed an isolated managed install, dependency check and actual application startup on macOS. Its checks used synthetic data and made no model requests. The other four updated Agent recipes have been resolved for their targets; current-release runtime checks on those systems remain pending.

`platform-matrix.json` records the recipe hashes, Python versions, notable dependency versions and runtime verification status. `validation.json` records the scope of completed runtime checks. Earlier release checks do not validate a newer Agent recipe.

## Install, validate and use

Select **Install & validate** to create a new environment, or enter an existing Python executable and select **Validate existing environment**. Then select **Use**. Existing environments and user configuration are preserved if installation or validation fails.

Validation checks HDF5 round trips, numerical operations and image handling. Core also exercises UMAP, video encoding, trajectory filtering and spreadsheet export. Agent validation runs the repository's actual `4_NLUI/agent.py --check` with a temporary session and image directory. It blocks network connections and supplies fake credentials; no provider requests are made.

Validation reports include a fingerprint of the relevant application code and dependency requirements. When these change, an older environment appears as **Revalidate** and cannot be activated or used for processing until it passes the current checks. This also prevents an old Gradio/LangChain environment from being treated as compatible with an updated Agent.

## Reproducible dependencies

Direct dependencies are in `core.in` and `agent.in`. Agent inherits the public application requirements rather than maintaining a separate API stack. Generated lock files pin the dependency closure and artifact hashes. Windows uses `core.lock` and `agent.lock`; other platforms use `PROFILE-PLATFORM.lock`. Installation accepts binary wheels only and never upgrades an existing environment in place.

The Core Intel Mac recipe pins Numba 0.61.2 for wheel compatibility. Core recipes retain their existing releases; the updated Agent release is `2026.11.03.1` on all five targets.

To regenerate an Agent recipe with uv 0.8.22 from the repository root:

```bash
uv pip compile ui/environments/agent.in \
  --python-version 3.12.11 --python-platform aarch64-apple-darwin \
  --generate-hashes --only-binary :all: --no-header \
  --output-file ui/environments/agent-macos-arm64.lock
```

Use the corresponding target and output filename for each platform, then update the hashes in `platform-matrix.json`. Run an actual installation and validation before recording runtime success for a target.

The UI downloads a fixed uv bootstrap wheel from PyPI, verifies its SHA-256, and installs managed Python into application storage. macOS/Linux executables are cached separately by platform. The environment's own Python path is retained rather than resolving its symlink to the base interpreter.

## Storage and remote use

macOS defaults to `~/Library/Application Support/MouseGPT`, retaining a pre-existing `~/.local/share/MouseGPT` installation. Linux honors `XDG_DATA_HOME`; `MOUSEGPT_UI_HOME` can select a different application directory. Windows uses the local application-data directory.

Installation requires internet access and at least 3 GB of free space, plus space for datasets. Processing and environments live on the machine hosting the UI. Cancelling an installation stops its process tree; a cancelled candidate is never activated.

The UI server still requires Python to start. This is a browser workbench, not a standalone desktop package. See the [UI user guide](../README.md) for the workflow and remote-access instructions.
