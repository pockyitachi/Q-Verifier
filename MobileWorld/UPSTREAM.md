# MobileWorld support for Q-Verifier

This directory is maintained inside `pockyitachi/Q-Verifier`. It is an ordinary
source directory, not a second Git repository or a submodule pointing to the
upstream MobileWorld repository.

- Upstream: https://github.com/Tongyi-MAI/MobileWorld
- Imported base commit: `2c43f6a6160fa37596a717194f2301e93fbed559`
- The upstream license is retained in `LICENSE`.
- `../docs/mobileworld-import.json` records the imported working files and their
  original SHA-256 digests. The import includes the local Q-Reflector changes;
  it is not an unchanged upstream checkout.

The maintained additions include MAI and GUI-Owl reflection feedback, shared
and v9 prompts, coordinate metadata, token accounting, environment recovery,
training and offline evaluation scripts, and CPU tests. The old standalone
layout placed Q-Verifier below MobileWorld. Dataset/model defaults in maintained
code address the parent Q-Verifier directory and its `training/` stages.

## Runtime assets

Model weights, datasets containing screenshot pairs, completed experiment
directories, Android virtual disks, API credentials, ADB private keys and
application database volumes are not imported into this source directory.
The Docker recipes require separately provisioned runtime assets, as they did
in the original workspace; this repository is not a complete emulator image.
Use the existing prepared MobileWorld image for the recorded experiments.
See the upstream setup guide for provisioning a new environment.

The three app-source submodules retain their upstream pinned commits under
`MobileWorld/resources/`; their declarations live in the repository-root
`.gitmodules`. From the repository root, initialize them only when needed:

```bash
git submodule update --init --recursive
```

## Existing server runs

The original experiment store is `../artifacts/`, relative to the repository
root. Those runs retain their original paths and frozen source snapshots.
This directory is the maintained source
for subsequent changes. Do not continue editing the old outer checkout and
expect those edits to appear in this repository automatically.
