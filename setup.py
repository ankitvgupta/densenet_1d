from setuptools import setup, find_packages

setup(name='densenet',
      version='0.5',
      description='An implementation of DenseNet for 1D inputs in Keras.',
      url='https://github.com/ankitvgupta/densenet_1d',
      author='Ankit Gupta',
      author_email='ankitgupta@alumni.harvard.edu',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
    # install_requires=['keras'],
      )
