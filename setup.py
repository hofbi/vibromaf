"""VibroMAF package configuration"""

from pathlib import Path

import setuptools

ROOT_DIR = Path(__file__).parent
README = ROOT_DIR / "README.md"
CHANGELOG = ROOT_DIR / "CHANGELOG.md"
REQUIREMENTS = ROOT_DIR / "requirements.txt"

setuptools.setup(
    name="vibromaf",
    version="0.0.3",
    author="Markus Hofbauer, Andreas Noll",
    author_email="name.surname@tum.de",
    description="Vibrotactile quality metrics and metric fusion",
    long_description=README.read_text() + "\n" + CHANGELOG.read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/hofbi/vibromaf",
    include_package_data=True,
    install_requires=REQUIREMENTS.read_text().split(),
    license="MIT",
    zip_safe=False,
    keywords="haptics, vibrotactile quality assessment, signal processing",
    project_urls={
        "Bug Tracker": "https://github.com/hofbi/vibromaf/issues",
        "Discussions": "https://github.com/hofbi/vibromaf/discussions",
        "Documentation": "https://hofbi.github.io/vibromaf",
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    package_data={
        "models": ["model/*"],
    },
    packages=setuptools.find_packages(include=["vibromaf.*"]),
    py_modules=["vibromaf.*"],
    python_requires=">=3.7",
)
