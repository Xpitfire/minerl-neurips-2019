from setuptools import find_packages, setup

readme = open("README.md").read()

requirements = {"install": ["ffmpeg", "PyHamcrest", "coloredlogs",
                            "tqdm", "natsort", "numpy", "torch", "torchvision",
                            "sklearn", "matplotlib", "seaborn", "pandas",
                            "gym", "imageio", "minerl", "keyboard", "minerl",
                            "natsort", "tensorboard", "petname", "pandas", "setproctitle",
                            "multiprocess", "python-box", "gym[Box_2D]", "box2d-py", "pyvirtualdisplay"]}
install_requires = requirements["install"]

setup(
    # Metadata
    name="MineRL",
    author="Xpitfire",
    version="0.0.1",
    author_email="dinu.marius-constantin@hotmail.com",
    url="https://github.com/Xpitfire/minerl-neurips-2019",
    description="Codebase for the NeurIPS MineRL Competition 2019",
    long_description=readme,
    long_description_content_type="text/markdown",
    # Package info
    packages=find_packages(),
    zip_safe=True,
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)
