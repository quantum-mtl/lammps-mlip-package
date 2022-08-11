# Documents

## Installation and compiling

```shell
pip install -r requirements.txt
sphinx-build docs docs_build
```

Auto build
```shell
sphinx-autobuild docs docs_build
# open localhost:8000 in your browser
```

## How to add references

- `references.bib` contains citation entries for [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.io/en/latest/index.html)
- Each subsection in `references.md` has a bibliography as follows
  - use `cite` directive to refer to an entry in `references.bib`
  - set `keyprefix` for each subsection and append it when refer to bibtex's entries.

```
    {cite}`LAMMPS`

    ```{bibliography}
    :filter: docname in docnames
    ```
```
