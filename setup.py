import setuptools

_name = 'mtlasso'

setuptools.setup(
  name=_name,
  description='Sparse multi-task lasso',
  version='0.1',
  url=f'https://www.github.com/aksarkar/{_name}',
  author='Abhishek Sarkar',
  author_email='aksarkar@uchicago.edu',
  license='MIT',
  install_requires=[
    'numpy',
  ],
  tests_require=[
    'pytest',
    'scikit-learn',
  ],
  packages=setuptools.find_packages('src'),
  package_dir={'': 'src'},
)
