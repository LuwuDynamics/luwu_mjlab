"""Installation script for the 'luwu_mjlab' python package."""

from setuptools import setup, find_packages

# Pin versions aligned with a known-good unitree_rl_mjlab stack.
# mjlab 1.2.0 uses wp.context.runtime; warp-lang>=1.14 removes that API.
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco==3.6.0",
    "mujoco-warp==3.6.0",
    "warp-lang==1.12.1",
    "scipy>=1.17.0",
]

# Installation operation
setup(
    name="luwu_mjlab",
    packages=["src"],
    version="1.0.0",
    install_requires=INSTALL_REQUIRES,
)
