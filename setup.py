#! /usr/bin/env python
"""
HoogBERTa: Multi-task Sequence Labeling using Thai Pretrained Language Representation
Copyright (C) July 2020 National Electronics and Computer Technology Center, Thailand. All rights reserved.
Written by Peerachet Porkaew
National Electronics and Computer Technology Center, Thailand
"""

from setuptools import setup

setup(
    name='hoogberta',
    packages=['hoogberta'],
    include_package_data=True,
    version='0.1.0',
    install_requires=['torch==1.8','fairseq==0.10.0','seqeval','subword-nmt','pytorch-lightning==1.4.7','scikit-learn','gdown',
               'torchtext==0.6.0','attacut'],
    license='MIT',
    description='HoogBERTa: Multi-task Sequence Labeling using Thai Pretrained Language Representation',
    author='Peerachet Porkaew',
    author_email='peerachet.porkaew@nectec.or.th',
    url='https://github.com/lstnlp/HoogBERTa',
    download_url='https://github.com/lstnlp/HoogBERTa/archive/refs/tags/v0.1.0.tar.gz',
    keywords=['HoogBERTa: Multi-task Sequence Labeling using Thai Pretrained Language Representation'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: General',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)