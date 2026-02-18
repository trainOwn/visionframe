from pathlib import Path
from setuptools import find_packages, setup

BASE_DIR = Path(__file__).parent
README = (BASE_DIR / "README.md").read_text(encoding="utf-8") if (BASE_DIR / "README.md").exists() else "drawcv"

if (BASE_DIR / "src").exists():
    setup(
        name="drawcv",
        version="0.1.2",
        description="Stylized OpenCV rectangle and frame drawing helpers for detection overlays.",
        long_description=README,
        long_description_content_type="text/markdown",
        python_requires=">=3.10",
        author="Kaushal",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        include_package_data=True,
        install_requires=[
            "numpy>=1.24",
            "opencv-python>=4.8",
        ],
        extras_require={
            "dev": ["build>=1.2", "twine>=5.1", "pytest>=8.0"],
        },
    )
else:
    setup(
        name="drawcv",
        version="0.1.2",
        description="Stylized OpenCV rectangle and frame drawing helpers for detection overlays.",
        long_description=README,
        long_description_content_type="text/markdown",
        python_requires=">=3.10",
        author="Kaushal",
        py_modules=["drawcv"],
        include_package_data=True,
        install_requires=[
            "numpy>=1.24",
            "opencv-python>=4.8",
        ],
        extras_require={
            "dev": ["build>=1.2", "twine>=5.1", "pytest>=8.0"],
        },
    )

