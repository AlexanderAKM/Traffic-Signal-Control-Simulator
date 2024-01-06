from setuptools import setup

with open('requirements.txt', 'r') as file:
    requirements = [line.strip() for line in file if line.strip()]

setup(
    name='FinalProject',
    description='Final Project of Reinforcement Learning Practical at University of Groningen',
    install_requires=requirements,
)
