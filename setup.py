import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="llm-mri",
    version="0.1.0",
    author="Luiz Felipe Costa",
    author_email="luizfelipecorradini@gmail.com",
    description="Package to visualize LLM's Neural Networks activation regions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luizcelsojr/LLM-MRI",
 
)