from setuptools import setup, find_packages

setup(
    name='memgpt',
    version='0.1.15',
    description='Memory Augmented GPT',
    author='Paul Krahn',
    packages=find_packages(where='.'),  # This will find packages in the current directory
    install_requires=[
        'pinecone-client',
        'openai',
        'tiktoken',
        'nltk',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  # Depending on the state of your project, this might be different
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',  # Update with your chosen license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
