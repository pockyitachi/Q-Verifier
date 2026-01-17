# Agent-SAMA on AndroidWorld

[![Paper](https://img.shields.io/badge/AAAI%202026-Paper-blue)](https://doi.org/10.5281/zenodo.15430188)
[![Agent-SAMA](https://img.shields.io/badge/Agent--SAMA-Main%20Repo-green)](../Agent-SAMA/README.md)
[![AndroidWorld](https://img.shields.io/badge/AndroidWorld-Benchmark-orange)](https://github.com/google-research/android_world)

> This guide explains how to run **Agent-SAMA** on the **AndroidWorld** benchmark using an Android emulator.

## 📖 Overview

[Agent-SAMA](../Agent-SAMA/README.md) is a state-aware multi-agent framework for autonomous mobile phone operation. This integration allows you to evaluate Agent-SAMA on AndroidWorld's 116 benchmark tasks.

### Key Differences from Original Agent-SAMA

| Aspect | Original Agent-SAMA | Agent-SAMA + AndroidWorld |
|--------|---------------------|---------------------------|
| **Device** | Physical Android phone | Android Emulator (AVD) |
| **Benchmarks** | Mobile-Eval, SPA-Bench | AndroidWorld (116 tasks) |
| **Connection** | USB + ADB | gRPC (port 8554) |
| **Evaluation** | Manual verification | Automatic reward signals |
| **Setup** | Phone required | Emulator only |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AndroidWorld Environment                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌──────────────────────────────────────────────────────┐     │
│    │              Agent-SAMA Adapter                       │     │
│    │  ┌────────────────────────────────────────────────┐  │     │
│    │  │  • Convert AndroidWorld UI → Agent-SAMA format │  │     │
│    │  │  • Convert Agent-SAMA actions → JSONAction     │  │     │
│    │  │  • Bridge evaluation interfaces                │  │     │
│    │  └────────────────────────────────────────────────┘  │     │
│    └──────────────────────────────────────────────────────┘     │
│                              │                                   │
│    ┌──────────────────────────────────────────────────────┐     │
│    │              Agent-SAMA Framework                     │     │
│    │                                                       │     │
│    │   ┌─────────┐  ┌─────────┐  ┌────────────┐          │     │
│    │   │ Planner │→ │  Actor  │→ │ Reflection │          │     │
│    │   └─────────┘  └─────────┘  └────────────┘          │     │
│    │        ↓                           ↓                 │     │
│    │   ┌─────────┐              ┌────────────┐           │     │
│    │   │  State  │              │   Mentor   │           │     │
│    │   │  Agent  │              │ (Learning) │           │     │
│    │   └─────────┘              └────────────┘           │     │
│    └──────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 💻 Prerequisites

- **Python 3.11+**
- **Android Studio** (for emulator)
- **API Keys**: OpenAI and/or Gemini
- **~10GB disk space** for emulator

## 🚀 Quick Start

### 1. Set Up Android Emulator

```bash
# Install Android Studio from:
# https://developer.android.com/studio

# Create AVD (Android Virtual Device):
# - Open Android Studio → Device Manager → Create Device
# - Hardware: Pixel 6
# - System Image: Tiramisu (API Level 33)
# - AVD Name: AndroidWorldAvd
```

### 2. Launch Emulator

```bash
# macOS
~/Library/Android/sdk/emulator/emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554

# Linux
~/Android/Sdk/emulator/emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554
```

### 3. Install Dependencies

```bash
cd android_world

# Create environment (recommended)
conda create -n android_world python=3.11.8
conda activate android_world

# Install AndroidWorld
pip install -r requirements.txt
python setup.py install
```

### 4. Set API Keys

```bash
# Add to ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="your-openai-key"
export QWEN_API_KEY="your-qwen-key"  # For Agent-SAMA perception
```

### 5. First Run (Install Apps)

```bash
# This installs required apps on the emulator (ONE TIME ONLY)
python run.py \
    --agent_name=agent_sama \
    --suite_family=android_world \
    --perform_emulator_setup \
    --tasks=ContactsAddContact
```

### 6. Run Benchmark

```bash
# Single task
python run.py \
    --agent_name=agent_sama \
    --suite_family=android_world \
    --tasks=ContactsAddContact

# Multiple tasks
python run.py \
    --agent_name=agent_sama \
    --suite_family=android_world \
    --tasks=ContactsAddContact,ClockStopWatchRunning,SettingsEnableWifi

# Full benchmark (all 116 tasks)
python run.py \
    --agent_name=agent_sama \
    --suite_family=android_world
```

## ⚙️ Command Line Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--agent_name` | Agent to use (`agent_sama`, `agent_sama_gemini`) | `m3a_gpt4v` |
| `--suite_family` | Task family (`android_world`, `miniwob`) | `android_world` |
| `--tasks` | Specific tasks (comma-separated) | All tasks |
| `--n_task_combinations` | Number of parameter variations per task | `1` |
| `--perform_emulator_setup` | Install apps on emulator (first time) | `False` |
| `--checkpoint_dir` | Resume from checkpoint | - |
| `--output_path` | Results output directory | `~/android_world/runs` |

## 📊 Available Agents

| Agent Name | Model | Description |
|------------|-------|-------------|
| `agent_sama` | GPT-4o | Agent-SAMA with OpenAI GPT-4o |
| `agent_sama_gemini` | Gemini 1.5 Pro | Agent-SAMA with Google Gemini |
| `m3a_gpt4v` | GPT-4V | Original AndroidWorld agent |
| `t3a_gpt4` | GPT-4 | Text-only agent |

## 📝 Example Tasks

### Simple Tasks (Good for Testing)

```bash
# Add a contact
--tasks=ContactsAddContact

# Start stopwatch
--tasks=ClockStopWatchRunning

# Toggle WiFi
--tasks=SettingsEnableWifi
```

### Cross-App Tasks

```bash
# Calendar + Contacts
--tasks=CalendarAddEventFromContacts

# Multiple apps
--tasks=SendSmsFromContacts,CalendarAddEvent
```

## 🔧 Minimal Python Example

```python
from android_world import registry
from android_world.env import env_launcher
from android_world.agents import agent_sama

# Initialize environment
env = env_launcher.load_and_setup_env(
    console_port=5554,
    emulator_setup=False,
    adb_path='~/Library/Android/sdk/platform-tools/adb',
)
env.reset(go_home=True)

# Get task
task_registry = registry.TaskRegistry()
task_type = task_registry.get_registry('android_world')['ContactsAddContact']
task = task_type(task_type.generate_random_params())
task.initialize_task(env)

# Create and run Agent-SAMA
agent = agent_sama.AgentSAMA(env, model_name='gpt-4o-2024-11-20')

print(f'Goal: {task.goal}')
for _ in range(int(task.complexity * 10)):
    response = agent.step(task.goal)
    if response.done:
        break

# Check result
success = task.is_successful(env) == 1
print(f'{"✅ Success" if success else "❌ Failed"}')

env.close()
```

## 📁 Project Structure

```
Agent-SAMA/
├── Agent-SAMA/                    # Main Agent-SAMA framework
│   ├── Agents/                    # Agent implementations
│   ├── AgentsUtils/               # Utilities
│   ├── run.py                     # Original runner (physical phone)
│   └── README.md                  # Main documentation
│
└── android_world/                 # AndroidWorld integration
    ├── android_world/
    │   └── agents/
    │       └── agent_sama.py      # Agent-SAMA adapter
    ├── run.py                     # AndroidWorld runner
    └── README_AGENT_SAMA.md       # This file
```

## 🔗 Related Links

- **[Agent-SAMA Main Repository](../Agent-SAMA/README.md)** - Full documentation
- **[AndroidWorld Paper](https://arxiv.org/abs/2405.14573)** - Benchmark details
- **[Agent-SAMA Paper (AAAI 2026)](https://doi.org/10.5281/zenodo.15430188)** - Framework details

## 📚 Citation

If you use Agent-SAMA on AndroidWorld, please cite both papers:

```bibtex
@inproceedings{guo2026agentsama,
  title={Agent-SAMA: State-Aware Mobile Assistant},
  author={Guo, Linqiang and Liu, Wei and Heng, Yi Wen and Chen, Tse-Hsun (Peter) and Wang, Yang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}

@misc{rawles2024androidworld,
  title={AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents},
  author={Christopher Rawles and others},
  year={2024},
  eprint={2405.14573},
  archivePrefix={arXiv}
}
```

## ❓ Troubleshooting

### Emulator not found
```bash
# Check if emulator is running
adb devices
# Should show: emulator-5554  device
```

### Agent-SAMA import error
```bash
# Ensure you're in the right directory
cd android_world
python -c "from android_world.agents import agent_sama; print('OK')"
```

### gRPC connection failed
```bash
# Make sure emulator was launched with -grpc 8554
# Restart emulator with correct flag
```

## 📄 License

This integration follows the licenses of both projects:
- Agent-SAMA: MIT License
- AndroidWorld: Apache 2.0 License

