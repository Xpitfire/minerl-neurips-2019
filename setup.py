from setuptools import find_packages, setup

readme = open("README.md").read()

requirements = {"install": ["ffmpeg", "PyHamcrest", "gym", "tensorboardX", "minerl"]}
install_requires = requirements["install"]

setup(
    # Metadata
    name="minerl_neurips_2019",
    author="MineRL 2019 competition Team",
    author_email="dinu.marius-constantin@hotmail.com",
    url="https://github.com/Xpitfire/minerl-neurips-2019",
    description="Codebase for the NeurIPS MineRL Competition 2019",
    long_description=readme,
    long_description_content_type="text/markdown",
    # Package info
    packages=find_packages(),
    zip_safe=True,
    install_requires=install_requires,
)