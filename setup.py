from setuptools import setup, find_packages

setup(
  name = 'coco-lm-pytorch',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'COCO - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/coco-lm-pytorch',
  keywords = [
    'transformers',
    'artificial intelligence',
    'deep learning',
    'pretraining'
  ],
  install_requires=[
    'torch>=1.6.0',
    'einops',
    'x-transformers'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)