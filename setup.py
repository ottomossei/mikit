from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

setup(
    name="mikit",
    version="1.0.0",
    description="Chemical formula-based materials informatics kit",
    author="Ottomossei",
    author_email="seki.jobhunting@gmail.com",
    install_requires=install_requirements,
    url='https://github.com/Ottomossei/mikit/',
    license=license,
    packages=find_packages(exclude=['example'])
)