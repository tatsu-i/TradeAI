from setuptools import setup, find_packages
import os

setup(
    name="csvdumper",
    version=0.1,
    python_requires="~=3.5",
    description="rabbitmq csv dumper",
    packages=find_packages(exclude="tests"),
    install_requires=[
        "aio-pika==4.9.3",
        "pika==1.1.0",
        "python-rapidjson",
        "elasticsearch",
        "pytz",
        "retry",
        "mackerel.client"
    ],
    extras_require={"test": ["pytest"]},
    scripts=[
        "dumper.py",
    ],
    include_package_data=True,
    zip_safe=False,
)
