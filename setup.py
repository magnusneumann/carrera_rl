from setuptools import setup, find_packages

setup(
    name="carrera_rl",
    version="0.1.0",
    description="RL Prototyping für Carrera Hybrid",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
