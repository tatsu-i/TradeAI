from setuptools import setup, find_packages
import os

setup(
    name="crawler",
    version=0.1,
    python_requires="~=3.5",
    description="yahoo finace crawler",
    packages=find_packages(exclude="tests"),
    install_requires=[
        "mackerel.client",
        "tqdm",
        "selenium",
        "elasticsearch"
    ],
    extras_require={"test": ["pytest"]},
    scripts=[
        "run.py",
        "store.py"
    ],
    include_package_data=True,
    zip_safe=False,
)
