# setup.py

from setuptools import setup, find_packages

setup(
    name='linelistSTAN',
    version='0.1.0',
    description='A description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/linelistSTAN',  # Update this with your URL
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        # Example: 'numpy>=1.18.5'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  # Choose your license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)
