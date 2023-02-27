import ast
import setuptools


def get_version(file_name: str, version_name: str = "__version__") -> str:
  """Find version by AST parsing to avoid needing to import this package."""
  with open(file_name) as f:
    tree = ast.parse(f.read())
    # Look for all assignment nodes in the AST, if the variable name is what
    # we assigned the version number too, grab the value (the version).
    for node in ast.walk(tree):
      if isinstance(node, ast.Assign):
        if node.targets[0].id == version_name:
          return node.value.s
  raise ValueError(f"Couldn't find assignment to variable {version_name} "
                   f"in file {file_name}")

_jax_version = "0.3.1"

setuptools.setup(
    name="t5x-demo",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        "": ["**/*.gin", "**/*.json"],
    },
    scripts=[],
    install_requires=[
        "absl-py",
        "flax @ git+https://github.com/google/flax#egg=flax",
        "gin-config",
        f"jax>={_jax_version}",
        "numpy",
        "seqio-nightly",
        "t5",
        "tensorflow",
        "tensorflow_datasets",
        "datasets",
        "promptsource",
        # Install from git as they have setup.pys but are not on PyPI.
        "t5x @ git+https://github.com/google-research/t5x.git@10da41308e756da0e68dfc12ededb18cc603714f",
    ],
    extras_require={
        "test": ["pytest>=6.0"],
        # TODO: mt5 and byt5 are not setup as python packages.
        # Figure out best way to bring them in as dependencies.
        "mt5": [],
        "byt5": [],
        "mrqa": ["pandas"],
        "tpu": [f"jax[tpu]>={_jax_version}"]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "prompt tuning",
        "machine learning",
        "transformers",
        "neural networks",
        "pre-trained language models",
        "nlp",
        "jax",
        "flax",
        "t5",
        "t5x",
    ]
)
