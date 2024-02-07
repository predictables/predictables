# PredicTables

## A brazenly-opinionated library for standardizing machine learning workflows in Python.

### Installation

(I mean, this is probably what it will look like, but I haven't actually published it yet, so...)

```bash
pip install predictables
```

Until then, can build my docker dev container (from root) to see where I am:

```Docker
docker build -t predictables-dev
docker run -it -p 4000:80 --name predictables-dev predictables-dev
```

This will build a container with Python 3.11 and install all necessary packages to run predictables (or at least enough to make all my tests run).

## Features

- **Automated Univariate Analysis:** Quick insights into each variable.
- **Correlation Matrix:** Understand how your features relate.
- **Principal Component Analysis (PCA):** Reduce dimensions, retain significance.
- **SHAP Analysis:** Interpret the impact of your features.
- **Bayesian Model Optimization:** Fine-tune models for peak performance.

## Installation

```bash
pip install predictables
```

## Quick Start

Get started with PredicTables in just a few lines:

```
import predictables as pt
pt.eda('path/to/your/data.csv')
```

## Contributing

PredicTables thrives on community contributions. Whether it’s improving code, fixing bugs, or enhancing documentation, every bit of help counts. See CONTRIBUTING.md for more details.

## License

PredicTables is released under the MIT License. See the LICENSE file for more details.
