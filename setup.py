#!/usr/bin/python
"""adenine setup script."""

from setuptools import setup

# Package Version
from secoc import __version__ as version

setup(
    name='secoc',
    version=version,

    description=('Sliding Window Error Correcting Code'),
    long_description=open('README.md').read(),
    author='Federico Tomasi, Samuele Fiorini',
    author_email='{federico.tomasi, samuele.fiorini}@dibris.unige.it',
    maintainer='Federico Tomasi, Samuele Fiorini',
    maintainer_email='{federico.tomasi, samuele.fiorini}@dibris.unige.it',
    url='https://github.com/slipguru/secoc',
    download_url='https://github.com/slipguru/secoc/tarball/'+version,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'License :: OSI Approved :: BSD License',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Informatics',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    license='FreeBSD',

    packages=['secoc'],
    install_requires=['numpy (>=1.10.1)',
                      'scipy (>=0.16.1)',
                      'scikit-learn (>=0.18)',
                      # 'matplotlib (>=1.5.1)',
                      #   'seaborn (>=0.7.0)',
                      #  'joblib',
                      #   'fastcluster (>=1.1.20)',
                      #   'GEOparse (>=0.1.10)',
                      #   'pydot (>=1.2.3)'
                      ],
    # scripts=['scripts/ade_run.py', 'scripts/ade_analysis.py',
    #          'scripts/ade_GEO2csv.py'
    #          ],
)
