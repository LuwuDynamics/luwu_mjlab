"""Installation script for the 'luwu_mjlab' python package."""

from setuptools import setup, find_packages

# Keep the simulation stack aligned with mjlab 1.6.
INSTALL_REQUIRES = [
    "mjlab==1.6.0",
    "mujoco==3.11.0",
    "mujoco-warp==3.11.0",
    "warp-lang==1.17.0",
    "scipy>=1.17.0",
]

# Installation operation
setup(
    name="luwu_mjlab",
    packages=["src"],
    version="1.0.0",
    install_requires=INSTALL_REQUIRES,
)
