import setuptools

with open("README.md","r",encoding="utf-8") as f:
    long_description = f.read()


__version__="0.0.0"
REPO_NAME="Fruits-and-vegetables-Image-classification"
AUTHOR_USER_NAME="sadam-hussain-s"
SRC_REPO="F&VClassifier"
AUTHOR_EMAIL="sadam89391@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker":f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src")
)