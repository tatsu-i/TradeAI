from setuptools import setup, find_packages
import os

setup(
    name="notifier",
    version=0.1,
    python_requires="~=3.5",
    description="notifier",
    packages=find_packages(exclude="tests"),
    install_requires=[
        "slackweb",
        "oauth2client",
        "google-api-python-client"
    ],
    extras_require={"test": ["pytest"]},
    scripts=[
        "run.py",
    ],
    include_package_data=True,
    zip_safe=False,
)
