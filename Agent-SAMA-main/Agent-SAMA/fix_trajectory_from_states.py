#!/usr/bin/env python3
"""
Fix from_state descriptions in existing trajectory data.

The problem: from_state was incorrectly capturing future state info instead of 
the state BEFORE the action.

The fix: For step N, from_state should be the to_state from step N-1.
For step 1, we use the initial screen state (inferred from context).

Usage:
    python fix_trajectory_from_states.py
    python fix_trajectory_from_states.py --dry-run  # Preview changes without modifying files
"""

import os
import json
import argparse
import shutil
from datetime import datetime


def fix_trajectory_file(filepath: str, dry_run: bool = False) -> dict:
    """
    Fix the from_state fields in a trajectory JSON file.
    
    For each transition N (N > 1):
    - from_state.description = to_state.description from transition N-1
    - from_state.precondition = to_state.precondition from transition N-1
    - from_state.postcondition = to_state.postcondition from transition N-1
    
    For transition 1:
    - We infer from the instruction and first action
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if 'transitions' not in data:
        return {"status": "skipped", "reason": "no transitions field"}
    
    transitions = data['transitions']
    if len(transitions) == 0:
        return {"status": "skipped", "reason": "empty transitions"}
    
    changes_made = 0
    
    for i, transition in enumerate(transitions):
        step_num = transition.get('step', i + 1)
        
        if i == 0:
            # For step 1: Use instruction context to describe initial state
            # The initial state is typically the home screen or app launcher
            instruction = data.get('instruction', '')
            action = transition.get('action', {})
            action_type = action.get('type', '')
            
            # Infer initial state from first action
            if action_type == 'Open_App':
                app_name = action.get('arguments', {}).get('app_name', 'an app')
                initial_description = f"The device home screen or app drawer is displayed, ready to open {app_name}."
                initial_precondition = "The device is unlocked and the home screen or app drawer is accessible."
                initial_postcondition = "The user can tap on app icons to launch applications."
            else:
                initial_description = "The initial screen state before the task begins."
                initial_precondition = "The device is ready for interaction."
                initial_postcondition = "The user can perform actions on the current screen."
            
            # Check if from_state needs fixing
            from_state = transition.get('from_state', {})
            if from_state.get('description') == 'Previous screen state' or not from_state.get('description'):
                transition['from_state'] = {
                    'description': initial_description,
                    'precondition': initial_precondition,
                    'postcondition': initial_postcondition
                }
                changes_made += 1
        else:
            # For step N > 1: Use to_state from previous transition
            prev_transition = transitions[i - 1]
            prev_to_state = prev_transition.get('to_state', {})
            
            # Get previous state info (without 'valid' field)
            new_from_state = {
                'description': prev_to_state.get('description', 'Previous screen state'),
                'precondition': prev_to_state.get('precondition', ''),
                'postcondition': prev_to_state.get('postcondition', '')
            }
            
            # Check if from_state needs fixing
            current_from_state = transition.get('from_state', {})
            if (current_from_state.get('description') == 'Previous screen state' or 
                not current_from_state.get('description') or
                current_from_state != new_from_state):
                transition['from_state'] = new_from_state
                changes_made += 1
    
    if changes_made > 0 and not dry_run:
        # Create backup
        backup_path = filepath + '.backup'
        if not os.path.exists(backup_path):
            shutil.copy(filepath, backup_path)
        
        # Write fixed data
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    return {
        "status": "fixed" if changes_made > 0 else "unchanged",
        "changes": changes_made,
        "total_steps": len(transitions)
    }


def fix_all_trajectories(trajectory_dir: str, dry_run: bool = False):
    """Fix all trajectory files in the given directory."""
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Fixing trajectory data in: {trajectory_dir}")
    print("=" * 60)
    
    stats = {
        "total_files": 0,
        "fixed_files": 0,
        "skipped_files": 0,
        "unchanged_files": 0,
        "total_changes": 0,
        "errors": []
    }
    
    # Find all trajectory JSON files
    for task_dir in sorted(os.listdir(trajectory_dir)):
        task_path = os.path.join(trajectory_dir, task_dir)
        if not os.path.isdir(task_path):
            continue
        
        # Look for trajectory_*.json files
        for filename in os.listdir(task_path):
            if filename.startswith('trajectory_') and filename.endswith('.json'):
                filepath = os.path.join(task_path, filename)
                stats["total_files"] += 1
                
                try:
                    result = fix_trajectory_file(filepath, dry_run)
                    
                    if result["status"] == "fixed":
                        stats["fixed_files"] += 1
                        stats["total_changes"] += result["changes"]
                        status_icon = "✅"
                    elif result["status"] == "skipped":
                        stats["skipped_files"] += 1
                        status_icon = "⏭️"
                    else:
                        stats["unchanged_files"] += 1
                        status_icon = "➖"
                    
                    print(f"{status_icon} {task_dir}: {result['status']} ({result.get('changes', 0)} changes)")
                    
                except Exception as e:
                    stats["errors"].append((task_dir, str(e)))
                    print(f"❌ {task_dir}: ERROR - {e}")
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Fixed: {stats['fixed_files']}")
    print(f"  Unchanged: {stats['unchanged_files']}")
    print(f"  Skipped: {stats['skipped_files']}")
    print(f"  Total changes: {stats['total_changes']}")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
        for task, error in stats["errors"]:
            print(f"    - {task}: {error}")
    
    if dry_run:
        print("\n[DRY RUN] No files were modified. Run without --dry-run to apply fixes.")
    else:
        print(f"\nBackups created with .backup extension")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Fix from_state descriptions in trajectory data')
    parser.add_argument('--trajectory-dir', '-d', 
                        default='trajectory_data',
                        help='Directory containing trajectory data (default: trajectory_data)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Preview changes without modifying files')
    parser.add_argument('--single', '-s', 
                        help='Fix a single trajectory file')
    
    args = parser.parse_args()
    
    if args.single:
        if os.path.exists(args.single):
            result = fix_trajectory_file(args.single, args.dry_run)
            print(f"Result: {result}")
        else:
            print(f"File not found: {args.single}")
    else:
        # Find trajectory_data directory
        if os.path.isabs(args.trajectory_dir):
            trajectory_dir = args.trajectory_dir
        else:
            # Try relative to script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            trajectory_dir = os.path.join(script_dir, args.trajectory_dir)
        
        if not os.path.exists(trajectory_dir):
            print(f"Trajectory directory not found: {trajectory_dir}")
            return
        
        fix_all_trajectories(trajectory_dir, args.dry_run)


if __name__ == '__main__':
    main()
