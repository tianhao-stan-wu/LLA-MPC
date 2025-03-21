from setuptools import setup

setup(
	name='llampc',
	version='1.0',
	author='Maitham AL-Sunni',
	author_email='malsunni@andrew.cmu.edu',
	install_requires=[
		'pandas==1.0.3',
		'cvxpy==1.1.24',
		'casadi==3.5.1',
		'botorch==0.1.4',
		'gpytorch==0.3.6',
		'matplotlib==3.1.2',
		'scikit-learn==0.22.2.post1',
		'tikzplotlib'
		],
	)
