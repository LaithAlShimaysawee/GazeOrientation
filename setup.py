import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='GazeOrientation',
    version='0.0.1',
    author='Laith Al-Shimaysawee',
    author_email='laith.alshimaysawee@gmail.com',
    description='installation of GazeOrientation package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/LaithAlShimaysawee/GazeOrientation',
    project_urls = {
        "Bug Tracker": "https://github.com/LaithAlShimaysawee/GazeOrientation/issues"
    },
    license='MIT',
    packages=['GazeOrientation'],
    install_requires=['numpy', 'opencv-python', 'mediapipe'],
) 