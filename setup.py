from setuptools import find_packages, setup


def read_requirements(file_name: str):
    """Lee requerimientos desde un archivo (requirements.txt) ignorando comentarios, etc."""
    with open(file_name, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            # Ignorar líneas vacías o que empiecen con #
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
        return lines


setup(
    name="latam_challenge",
    version="0.1.0",
    description="A flight-delay model and API",
    packages=find_packages(exclude=["tests*"]),
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.9",
)
