# Contribution Guidelines

## Setup

```shell
# Setup the python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements-dev.txt

# Install vibromaf as development (editable) package
pip3 install -e .
```

## Development

We use [pre-commit](https://pre-commit.com/) to manage our git pre-commit hooks.
`pre-commit` is automatically installed from `requirements-dev.txt`.
To set it up, call

```sh
git config --unset-all core.hooksPath # may fail if you don't have any hooks set, but that's ok
pre-commit install --overwrite
```

### Hooks Usage

With `pre-commit`, you don't use your linters/formatters directly anymore, but through `pre-commit`:

```sh
pre-commit run --file path/to/file1.cpp tools/second_file.py  # run on specific file(s)
pre-commit run --all-files  # run on all files tracked by git
pre-commit run --from-ref origin/master --to-ref HEAD  # run on all files changed on current branch, compared to master
pre-commit run <hook_id> --file <path_to_file>  # run specific hook on specific file
```
