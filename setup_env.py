#!/usr/bin/env python3
"""
Setup script for Gait Abnormality Detection System development environment.
This script creates a virtual environment and installs all required dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return None

def main():
    """Main setup function."""
    print("🚀 Setting up Gait Abnormality Detection System development environment")
    
    # Check if Python 3.8+ is available
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Create virtual environment
    venv_path = Path("venv")
    if not venv_path.exists():
        if run_command("python -m venv venv", "Creating virtual environment"):
            print("✓ Virtual environment created at ./venv")
        else:
            print("❌ Failed to create virtual environment")
            sys.exit(1)
    else:
        print("✓ Virtual environment already exists")
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_command = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_script = "source venv/bin/activate"
        pip_command = "venv/bin/pip"
    
    # Upgrade pip
    run_command(f"{pip_command} install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if Path("requirements.txt").exists():
        run_command(f"{pip_command} install -r requirements.txt", "Installing dependencies")
    else:
        print("❌ requirements.txt not found")
        sys.exit(1)
    
    # Install package in development mode
    run_command(f"{pip_command} install -e .", "Installing package in development mode")
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo activate the virtual environment:")
    if os.name == 'nt':
        print("  venv\\Scripts\\activate")
    else:
        print("  source venv/bin/activate")
    
    print("\nTo start Jupyter Lab:")
    print("  jupyter lab")
    
    print("\nTo run tests:")
    print("  pytest tests/")

if __name__ == "__main__":
    main()