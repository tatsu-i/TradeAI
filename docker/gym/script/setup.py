from setuptools import setup, find_packages
import os

setup(
    name="stock-gym",
    version=0.1,
    python_requires="~=3.5",
    description="stock-gym",
    packages=find_packages(exclude="tests"),
    install_requires=[
        "aio-pika==4.9.3",
        "pika==1.1.0",
        "python-rapidjson",
        "click",
        "redisai==0.4.0",
        "ml2rt==0.1.1",
        "retry",
        "elasticsearch_dsl",
    ],
    extras_require={"test": ["pytest"]},
    scripts=[
        "run.py",
        "train.py",
        "deploy.py",
        "import-optimization.py",
        "aggs-models.py",
    ],
    include_package_data=True,
    zip_safe=False,
)
