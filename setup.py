from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    req_path = Path(__file__).with_name("requirements.txt")
    requirements: list[str] = []
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


setup(
    name="minitorch",
    version="0.4",
    packages=find_packages(),
    install_requires=read_requirements(),
)
