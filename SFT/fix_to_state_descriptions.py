#!/usr/bin/env python3
"""
Fix to_state descriptions in existing trajectory data.

The problem: to_state.description was recorded as "Current screen state" placeholder
because the StateAgent hadn't analyzed the new screen yet at recording time.

The fix: For step N, to_state should be the from_state of step N+1.
For the last step, we keep the existing (or use a generic "Final screen state").

Usage:
    python fix_to_state_descriptions.py --trajectory-dir trajectory_data
    python fix_to_state_descriptions.py --trajectory-dir trajectory_data --dry-run
"""

import os
import json
import argparse
import shutil


def fix_trajectory_file(filepath: str, dry_run: bool = False) -> dict:
    """
    Fix the to_state fields in a trajectory JSON file.
    
    For each transition N (where N < last step):
    - to_state.description = from_state.description from transition N+1
    - to_state.precondition = from_state.precondition from transition N+1
    - to_state.postcondition = from_state.postcondition from transition N+1
    
    For the last transition:
    - Keep existing or use "Final screen state after completing the task"
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
        current_to_state = transition.get('to_state', {})
        current_desc = current_to_state.get('description', '')
        
        # Check if this needs fixing (has placeholder value)
        if current_desc in ["Current screen state", "", "Current screen state."]:
            if i < len(transitions) - 1:
                # Use from_state of the NEXT transition
                next_transition = transitions[i + 1]
                next_from_state = next_transition.get('from_state', {})
                
                # Copy from next step's from_state (without 'valid' field if present)
                new_to_state = {
                    'description': next_from_state.get('description', 'Screen state after action'),
                    'precondition': next_from_state.get('precondition', current_to_state.get('precondition', '')),
                    'postcondition': next_from_state.get('postcondition', current_to_state.get('postcondition', ''))
                }
                
                # Preserve 'valid' field if it exists
                if 'valid' in current_to_state:
                    new_to_state['valid'] = current_to_state['valid']
                
                # Only count as change if description actually changed
                if new_to_state['description'] != current_desc:
                    transition['to_state'] = new_to_state
                    changes_made += 1
            else:
                # Last step - use postcondition context or generic description
                postcond = current_to_state.get('postcondition', '')
                if postcond:
                    new_desc = f"Final state: {postcond}"
                else:
                    new_desc = "Final screen state after the last action"
                
                transition['to_state']['description'] = new_desc
                changes_made += 1
    
    if changes_made > 0 and not dry_run:
        # Create backup
        backup_path = filepath + '.to_state_backup'
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
    
    print(f"{'[DRY RUN] ' if dry_run else ''}Fixing to_state descriptions in: {trajectory_dir}")
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
                    
                    print(f"{status_icon} {task_dir}: {result['status']} ({result.get('changes', 0)}/{result.get('total_steps', 0)} steps fixed)")
                    
                except Exception as e:
                    stats["errors"].append((task_dir, str(e)))
                    print(f"❌ {task_dir}: ERROR - {e}")
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Fixed: {stats['fixed_files']}")
    print(f"  Unchanged: {stats['unchanged_files']}")
    print(f"  Skipped: {stats['skipped_files']}")
    print(f"  Total to_state changes: {stats['total_changes']}")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
        for task, error in stats["errors"]:
            print(f"    - {task}: {error}")
    
    if dry_run:
        print("\n[DRY RUN] No files were modified. Run without --dry-run to apply fixes.")
    else:
        print(f"\nBackups created with .to_state_backup extension")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Fix to_state descriptions in trajectory data')
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

