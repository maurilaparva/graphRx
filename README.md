
<!-- Improved compatibility of back to top link -->

<a id="readme-top"></a>

<!-- PROJECT LOGO -->

<br />
<div align="center">
  <img src="images/logo.png" alt="Drug Repurposing Discovery Logo" width="200" height="200">
  <h3 align="center">Drug Repurposing Discovery with Graph Neural Nets</h3>
  <p align="center">
    A project to identify potential drug-disease interactions using Graph Neural Networks on the DRKG dataset.
  </p>
</div>

## About The Project

The Drug Repurposing Discovery project utilizes graph neural networks to predict drug-disease interactions by leveraging the Drug Repurposing Knowledge Graph (DRKG). This project focuses on using a computationally efficient SIGN encoder and a quadratic decoder to improve link prediction tasks, enabling drug repurposing discovery with high accuracy.

### Built With

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [DGL (Deep Graph Library)](https://www.dgl.ai/)
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Getting Started

### Prerequisites

* Python 3.8+
* pip or conda

### Installation

1. Clone the repo

   ```bash
   git clone https://github.com/maurilaparva/drug-repurposing-gnn.git
   cd drug-repurposing-gnn
````

2. Download the DRKG dataset from the [GitHub repository](https://github.com/gnn4dr/DRKG) and place the following files in the `data/` directory:

   * `drkg.tsv`
   * `entities.tsv`
   * `relations.tsv`
   * `embed/` (folder containing `entities.tsv`, `relations.tsv`, etc.)

3. Create a virtual environment & install dependencies

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Usage

### Data Preprocessing & Feature Engineering

Run the preprocessing and feature extraction scripts:

```bash
python data_cleaning.py
python feature_engineering.py
```

### Train the Model

Run the model training script:

```bash
python train_model.py
```

### Evaluate the Model

Evaluate the trained model on the test set:

```bash
python evaluate_model.py
```

All necessary files and data should be placed in:

```
data/
├── drkg.tsv
├── entities.tsv
├── relations.tsv
└── embed/
    ├── entities.tsv
    ├── relations.tsv
    ├── ...
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Acknowledgments

* [DRKG - Drug Repurposing Knowledge Graph](https://github.com/gnn4dr/DRKG)
* [Graph Neural Networks for Drug Repurposing (Doshi et al., 2022)](https://doi.org/10.1016/j.compbiomed.2022.105992)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS -->

[contributors-shield]: https://img.shields.io/github/contributors/maurilaparva/drug-repurposing-gnn.svg?style=for-the-badge
[contributors-url]: https://github.com/maurilaparva/drug-repurposing-gnn/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/maurilaparva/drug-repurposing-gnn.svg?style=for-the-badge
[forks-url]: https://github.com/maurilaparva/drug-repurposing-gnn/network/members
[stars-shield]: https://img.shields.io/github/stars/maurilaparva/drug-repurposing-gnn.svg?style=for-the-badge
[stars-url]: https://github.com/maurilaparva/drug-repurposing-gnn/stargazers
[issues-shield]: https://img.shields.io/github/issues/maurilaparva/drug-repurposing-gnn.svg?style=for-the-badge
[issues-url]: https://github.com/maurilaparva/drug-repurposing-gnn/issues
[license-shield]: https://img.shields.io/github/license/maurilaparva/drug-repurposing-gnn.svg?style=for-the-badge
[license-url]: https://github.com/maurilaparva/drug-repurposing-gnn/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/mauriciovillavicencio

