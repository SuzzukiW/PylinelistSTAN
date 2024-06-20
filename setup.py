# setup.py

from setuptools import setup, find_packages

setup(
    name='linelistSTAN',
    version='0.2.0',
    description='COVID-19 case onset prediction package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Xiang Fu',
    author_email='xfu@bu.edu',
    url='https://github.com/SuzzukiW/PylinelistSTAN',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'pymc3',
        'theano',
        'matplotlib',
        'seaborn',
        'arviz'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)
