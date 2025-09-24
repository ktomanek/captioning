from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
      requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="captioning_lib",
    version="0.1",
    description="A library for captioning with pseudo streaming for several ASR models.",
    author="Katrin Tomanek",
    author_email="katryn.tomanek@gmail.com",    
    url="https://github.com/ktomanek/captioning",
    package_dir={"": "src"},  # Look in src/ directory
    packages=find_packages(where="src"),  # Find packages in src/    
    install_requires=requirements,
    python_requires=">=3.10",
)
