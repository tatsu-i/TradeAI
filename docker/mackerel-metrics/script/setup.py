from setuptools import setup, find_packages
import os

setup(
    name="mackerel-metrics",
    version=0.1,
    python_requires="~=3.5",
    description="mackerel metrics",
    packages=find_packages(exclude="tests"),
    install_requires=[
        "click",
        "pyvmomi",
        "mackerel.client"
    ],
    extras_require={"test": ["pytest"]},
    scripts=[
        "run.py",
    ],
    include_package_data=True,
    zip_safe=False,
)
