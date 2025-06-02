"""
Build script for production deployment.
"""
import os
import shutil
from pathlib import Path

def build():
    """Build the application for production deployment."""
    # Create dist directory
    dist_dir = Path("dist")
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir()

    # Copy static assets
    assets_dir = Path("assets")
    if assets_dir.exists():
        shutil.copytree(assets_dir, dist_dir / "assets")

    # Copy data files
    data_dir = Path("data")
    if data_dir.exists():
        shutil.copytree(data_dir, dist_dir / "data")

    # Copy Python files
    shutil.copy("app.py", dist_dir)
    shutil.copytree("src", dist_dir / "src")

    # Create runtime.txt for Python version
    with open(dist_dir / "runtime.txt", "w") as f:
        f.write("python-3.9")

    # Copy requirements.txt
    shutil.copy("requirements.txt", dist_dir)

    print("Build completed successfully!")

if __name__ == "__main__":
    build()
