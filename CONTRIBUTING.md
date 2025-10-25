# Collaboration Guidelines

Please follow these few rules so we can work smoothly.

## 0. Setup
See `README.md`.

## 1. Jupytext

We leverage [Jupytext](https://jupytext.readthedocs.io/en/latest/) to avoid Jupyter merge conflict hell.

That means, to work on the Jupyter notebook, use the linked `.py` file by following these steps.

1. Start Jupyter server in project root directory:
    ```bash
    jupyter notebook
    ```
2. In Jupyter, instead of opening the `.ipynb` file, right click on the linked `.py` file and select "Open With" -> "Jupyter Notebook" (you might have to explicitly click on the pop-up depending on your browser).
3. Only commit the linked `.py` file, not the `.ipynb` file.

## 2. Branching
- **Do not commit directly to `main`.**
- Create your own branch for any work:
    ```bash
    git switch -c yourname/feature-name
    ```
- When finished, open a ***pull request on GitHub*** into `main`, so we can review and discuss changes.

## 3. Before Committing
To keep our code style consistent, format files before committing or opening a PR.

For `.py` files that are ***NOT*** linked to a notebook:

```bash
black --line-length 119 <script_file.py>
```

For `.ipynb` files (via Jupytext):
```bash
jupytext --pipe "black --line-length 119 -" <notebook_name.ipynb>
```
