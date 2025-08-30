# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="yolov8-license-plate-recognition",
    version="2.0.0",
    author="License Plate Recognition Team",
    author_email="team@example.com",
    description="YOLOv8 기반 차량 번호판 인식 시스템",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/yolov8-license-plate-recognition",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
            "sphinx>=4.0.0",
        ],
        "gpu": [
            "torch[cuda]>=1.9.0",
            "torchvision[cuda]>=0.10.0",
        ],
        "postgresql": [
            "psycopg2-binary>=2.9.0",
        ],
        "redis": [
            "redis>=4.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "lpr-server=main_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)