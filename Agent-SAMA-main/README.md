# Agent-SAMA: State-Aware Mobile Assistant

[![Paper](https://img.shields.io/badge/AAAI%202026-Paper-blue)](https://doi.org/10.5281/zenodo.15430188)
[![Zenodo](https://img.shields.io/badge/Zenodo-Code-green)](https://zenodo.org/records/15430188)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Agent-SAMA** is a state-aware multi-agent framework that models mobile app execution as a **Finite State Machine (FSM)**, treating UI screens as states and user actions as transitions.

## 📖 Abstract

Mobile Graphical User Interface (GUI) agents aim to autonomously complete tasks within or across apps based on user instructions. While recent Multimodal Large Language Models (MLLMs) enable these agents to interpret UI screens and perform actions, existing agents remain fundamentally reactive—they reason over the current UI screen but lack a structured representation of the app navigation flow.

**Agent-SAMA** addresses this limitation by implementing four specialized agents that collaboratively construct and use FSMs in real time to guide:
- **Task Planning**: Decomposing instructions into executable subgoals
- **Execution Verification**: Comparing actual vs. expected state transitions
- **Error Recovery**: Using the FSM structure to navigate back to known states

### 📊 Results

| Benchmark | Success Rate | Recovery Rate |
|-----------|--------------|---------------|
| Mobile-Eval-E | **84.0%** | **71.9%** |
| SPA-Bench | **80.0%** | **66.7%** |
| AndroidWorld | **63.7%** | - |

Compared to prior methods, Agent-SAMA improves task success by up to **12%** and recovery success by **13.8%**.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent-SAMA Framework                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Planner    │───▶│    Actor     │───▶│  Reflection  │       │
│  │    Agent     │    │    Agent     │    │    Agent     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      InfoPool                            │    │
│  │  (Central State: FSM Graphs, Action History, Tips...)   │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    State     │    │   Screen     │    │    Mentor    │       │
│  │    Agent     │    │   Parser     │    │    Agent     │       │
│  │   (FSM)      │    │   (Vision)   │    │ (Evolution)  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | Role | Key Function |
|-------|------|--------------|
| **Planner** | Task decomposition | Generates subgoals, uses LLM-as-Judge for plan selection |
| **Actor** | Action execution | Executes atomic actions and learned shortcuts |
| **Reflection** | Verification | Checks action outcomes (A/B/C), tracks progress |
| **State Agent** | FSM modeling | Maintains state graphs with beacons, pre/post-conditions |
| **Screen Parser** | Perception | OCR + GroundingDINO + icon captioning |
| **Mentor** | Self-evolution | Generates reusable tips and shortcuts |

### FSM State Representation

Each UI screen is modeled as an FSM state with:
- **Beacon**: Unique identifier (e.g., "Search Page of Maps")
- **Pre-condition**: What must be true before entering this state
- **Post-condition**: What should be true after completing actions
- **Transitions**: Actions that lead to other states

## 💻 Environment Setup

> ❗ This framework is designed for **Android OS** only.  
> ❗ All experiments were conducted on a **Google Pixel 7 Pro** device.

### Installation

```bash
cd Agent-SAMA
pip install -r requirements.txt
```

### Connecting Your Mobile Device via ADB

1. **Install Android Debug Bridge (ADB)**:  
   Download the [Android platform tools](https://developer.android.com/tools/releases/platform-tools)

2. **Enable Developer Options** on your Android phone:  
   `Settings > About phone` → Tap **Build number** 7 times

3. **Enable USB Debugging**:  
   `Settings > Developer options` → Enable **USB debugging**

4. **Connect via USB** and select "Transfer files" mode

5. **Verify connection**:
   ```bash
   /path/to/adb devices
   ```

### Installing ADB Keyboard

1. Download [ADB Keyboard APK](https://github.com/senzhk/adbkeyboard/blob/master/adbkeyboard.apk)
2. Install on your device
3. Set as default input method: `Settings > Language & Input > ADB Keyboard`

## ⚙️ Configuration

Set the following environment variables:

```bash
# ADB Path
export ADB_PATH="/path/to/adb"

# Backbone Model (choose one)
export BACKBONE_TYPE="OpenAI"  # or "Gemini" or "Claude"

# API Keys (set the one matching your backbone)
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
export CLAUDE_API_KEY="your-claude-key"

# Screen Parser (icon captioning via Qwen API)
export QWEN_API_KEY="your-qwen-api-key"
```

### Supported Backbone Models

| Provider | Model |
|----------|-------|
| OpenAI | `gpt-4o-2024-11-20` |
| Gemini | `gemini-1.5-pro-latest` |
| Claude | `claude-3-5-sonnet-20241022` |

## 🚀 Quick Start

### Single Task

```bash
python run.py \
    --run_name "my_test" \
    --setting "single" \
    --instruction "Find the best-rated Korean restaurant in Champaign, IL on Google Maps." \
    --max_itr 50
```

### Multiple Tasks (with self-evolution)

```bash
python run.py \
    --run_name "multi_test" \
    --setting "multi" \
    --tasks_json "Data/Mobile-Eval-E/scenario_1_v1.json"
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--setting` | `single` or `multi` task mode | `single` |
| `--instruction` | Task instruction (single mode) | - |
| `--tasks_json` | Path to task JSON file (multi mode) | - |
| `--max_itr` | Maximum iterations per task | 40 |
| `--max_consecutive_failures` | Stop after N consecutive failures | 5 |
| `--max_repetitive_actions` | Stop after N identical actions | 5 |
| `--temperature` | LLM temperature | 0.0 |

## 📁 Project Structure

```
Agent-SAMA/
├── Agents/
│   ├── base_agent.py       # Abstract base class for all agents
│   ├── info_pool.py        # Central state repository
│   ├── planner.py          # Task decomposition & LLM-as-Judge
│   ├── actor.py            # Action selection & execution
│   ├── reflection_agent.py # Outcome verification & note-taking
│   ├── state_agent.py      # FSM state graph management
│   ├── mentor.py           # Self-evolution (tips & shortcuts)
│   └── screen_parser.py    # Visual perception pipeline
├── AgentsUtils/
│   ├── chat_utils.py       # LLM API wrappers
│   ├── common_utils.py     # JSON parsing, graph serialization
│   ├── ocr_utils.py        # OCR & icon detection
│   └── tool_utils.py       # ADB commands (tap, swipe, type)
├── Data/
│   ├── Mobile-Eval-E/      # Cross-app benchmark scenarios
│   └── SPA-Bench/          # Single/cross-app benchmark scenarios
├── constant.py             # Configuration constants
├── infer.py                # Main inference loop
├── run.py                  # Entry point
└── requirements.txt        # Dependencies
```

## 📚 Citation

If you use Agent-SAMA in your research, please cite:

```bibtex
@inproceedings{guo2026agentsama,
  title={Agent-SAMA: State-Aware Mobile Assistant},
  author={Guo, Linqiang and Liu, Wei and Heng, Yi Wen and Chen, Tse-Hsun (Peter) and Wang, Yang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ModelScope](https://modelscope.cn/) for OCR and GroundingDINO models
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) for icon captioning
- [NetworkX](https://networkx.org/) for FSM graph implementation
