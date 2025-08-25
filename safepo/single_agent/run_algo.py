#!/usr/bin/env python3
"""
Universal algorithm runner for SafePO single agent algorithms.

This program allows running any algorithm from the single_agent directory
while preserving all command line arguments that individual algorithms expect.

Usage:
    python run_algo.py --algo <algorithm> [other_args...]
    
Examples:
    python run_algo.py --algo ppo --task SafetyPointGoal1-v0 --seed 0
    python run_algo.py --algo cpo --task SafetyPointGoal1-v0 --seed 0 --cost-limit 25.0
    python run_algo.py --algo trpo --task SafetyPointGoal1-v0 --seed 0 --num-envs 20
"""

import sys
import os
import argparse
import importlib.util
import subprocess
from pathlib import Path
from typing import List, Optional


def get_available_algorithms() -> List[str]:
    """
    Get list of available algorithms from the current directory.
    
    Returns:
        List of algorithm names (without .py extension)
    """
    current_dir = Path(__file__).parent
    algorithms = []
    
    for file in current_dir.glob("*.py"):
        # Skip utility files and this file itself
        if (file.name not in ["__init__.py", "utils.py", "run_algo.py", "benchmark.py", "plot.py"] and 
            not file.name.startswith("_")):
            algorithms.append(file.stem)
    
    return sorted(algorithms)


def run_algorithm_as_module(algorithm: str, args: List[str]) -> bool:
    """
    Run the algorithm by importing it as a module and calling its main function.
    
    Args:
        algorithm: Name of the algorithm to run
        args: List of command line arguments to pass to the algorithm
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Add the parent directory to Python path to handle relative imports
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        # Construct the module path
        module_path = f"single_agent.{algorithm}"
        
        # Import the module
        spec = importlib.util.find_spec(module_path)
        if spec is None:
            raise ImportError(f"Could not find module {module_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Import the argument parser
        from utils.config import single_agent_args
        
        # Temporarily replace sys.argv to exclude --algo argument
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]] + args
        
        try:
            # Parse arguments
            parsed_args, cfg_env = single_agent_args()
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
        
        # Call the main function
        if hasattr(module, 'main'):
            module.main(parsed_args, cfg_env)
        else:
            raise AttributeError(f"Module {algorithm} does not have a main function")
            
    except Exception as e:
        print(f"Error running {algorithm} as module: {e}")
        return False
    
    return True


def run_algorithm_as_script(algorithm: str, args: List[str]) -> bool:
    """
    Run the algorithm as a separate Python script.
    
    Args:
        algorithm: Name of the algorithm to run
        args: List of command line arguments to pass to the algorithm
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Construct the script path
        script_path = Path(__file__).parent / f"{algorithm}.py"
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        # Run the script with the same arguments
        cmd = [sys.executable, str(script_path)] + args
        
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"Error running {algorithm} as script: {e}")
        return False
    except Exception as e:
        print(f"Error running {algorithm} as script: {e}")
        return False


def validate_algorithm(algorithm: str) -> bool:
    """
    Validate that the specified algorithm exists.
    
    Args:
        algorithm: Name of the algorithm to validate
        
    Returns:
        True if algorithm exists, False otherwise
    """
    available_algorithms = get_available_algorithms()
    return algorithm in available_algorithms


def print_algorithm_help(algorithm: str) -> None:
    """
    Print help information for a specific algorithm.
    
    Args:
        algorithm: Name of the algorithm to get help for
    """
    try:
        script_path = Path(__file__).parent / f"{algorithm}.py"
        if script_path.exists():
            cmd = [sys.executable, str(script_path), "--help"]
            subprocess.run(cmd, check=True)
        else:
            print(f"Algorithm {algorithm} not found")
    except Exception as e:
        print(f"Error getting help for {algorithm}: {e}")


def main():
    """Main function to run the algorithm runner."""
    # Get available algorithms
    available_algorithms = get_available_algorithms()
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Universal algorithm runner for SafePO single agent algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available algorithms: {", ".join(available_algorithms)}

Examples:
  python run_algo.py --algo ppo --task SafetyPointGoal1-v0 --seed 0
  python run_algo.py --algo cpo --task SafetyPointGoal1-v0 --seed 0 --cost-limit 25.0
  python run_algo.py --algo trpo --task SafetyPointGoal1-v0 --seed 0 --num-envs 20
  python run_algo.py --algo ppo --help  # Get help for specific algorithm
        """
    )
    
    # Add the algo argument
    parser.add_argument(
        '--algo', 
        type=str, 
        required=True,
        choices=available_algorithms,
        help=f'Algorithm to run. Available: {", ".join(available_algorithms)}'
    )
    
    # Add help flag for specific algorithm
    parser.add_argument(
        '--help-algo',
        action='store_true',
        help='Show help for the specified algorithm'
    )
    
    # Parse known arguments to get the algo and help flag
    args, remaining = parser.parse_known_args()
    
    algorithm = args.algo
    
    # Validate algorithm
    if not validate_algorithm(algorithm):
        print(f"Error: Algorithm '{algorithm}' not found.")
        print(f"Available algorithms: {', '.join(available_algorithms)}")
        sys.exit(1)
    
    # Handle help request for specific algorithm
    if args.help_algo:
        print(f"Help for algorithm: {algorithm}")
        print("=" * 50)
        print_algorithm_help(algorithm)
        return
    
    print(f"Running algorithm: {algorithm}")
    print(f"Arguments: {remaining}")
    print("-" * 50)
    
    # Try running as module first (more efficient)
    print("Attempting to run as module...")
    if run_algorithm_as_module(algorithm, remaining):
        print(f"Successfully ran {algorithm} as module")
        return
    
    # Fall back to running as script
    print("Falling back to running as script...")
    if run_algorithm_as_script(algorithm, remaining):
        print(f"Successfully ran {algorithm} as script")
        return
    
    print(f"Failed to run algorithm {algorithm}")
    sys.exit(1)


if __name__ == "__main__":
    main()
