import setuptools

# use readme as long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# read in required packages from requirements.txt
with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

#get version number
from subprocess import Popen, PIPE

try:
    process = Popen(['git', 'describe', '--exact-match', '--tags'], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    version = stdout.decode('ascii')
except:
    version = '0.0.0'

# set up setuptools
setuptools.setup(
    name="platypos",
    version="1.1.16",
    author='Laura Ketzer',
    author_email='lketzer@aip.de',
    description='PLAneTarY PhOtoevaporation Simulator',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lketzer/platypos",
    install_requires=requirements,
    keywords='astronomy, exoplanets, ...',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'Development Status :: 3 - Alpha', # 2 - Pre-Alpha, # 3 - Alpha, 4 - Beta, 5 - Production/Stable, 6 - Mature, 7 - Inactive (1 - Planning)
        'Environment :: Console',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

