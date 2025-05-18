from setuptools import setup, find_packages

setup(
    name="attention-is-all-you-need",
    version="0.1.0",
    description="Implementation of the Transformer model from 'Attention Is All You Need', for machine translation from English to German",
    author="Bhargav Borah",
    url="https://github.com/bhargav-borah/attention-is-all-you-need",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "pandas>=1.1.0",
        "tiktoken>=0.3.1",
        "nltk>=3.5",
        "tqdm>=4.50.0",
    ],
    python_requires=">=3.7",
)