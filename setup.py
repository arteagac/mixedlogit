import setuptools

setuptools.setup(name='mixedlogit',
      version='0.0.1',
      description='Estimation of mixed, multinomial, and conditional logit models in Python',
      long_description = 'Estimation of mixed, multinomial, and conditional logit models in Python',
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
