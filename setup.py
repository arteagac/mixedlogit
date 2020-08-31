import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='mixedlogit',
      version='0.0.2rc1',
      description='Estimation of mixed, multinomial, and conditional logit models in Python',
      long_description = long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/arteagac/mixedlogit',
      author='Cristian Arteaga',
      author_email='cristiandavidarteaga@gmail.com',
      license='MIT',
      packages=['mixedlogit'],
      zip_safe=False,
      python_requires='>=3.5',
      install_requires=[
          'numpy>=1.13.1',
          'scipy>=1.0.0'
      ])
