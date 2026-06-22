# Developers

<!-- toc -->

## Cloning the GitHub Repo

Clone the repository:

1. [Download and install git](https://git-scm.com/downloads) -- accept all defaults when installing.
1. Create a parent directory in your preferred location to contain the repo -- e.g.
   `<USER_HOME>/Documents/altrios_project/`.
1. Open git bash, and inside the directory you created,
   [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
   the [ALTRIOS repository](https://github.com/NREL/ALTRIOS) with e.g. `git clone
https://github.com/NREL/ALTRIOS.git`.

## Installing the Python Package

Within the ALTRIOS folder, run `pip install -e ".[dev]"`

## Rust Installation

Install Rust: <https://www.rust-lang.org/tools/install>.

## Automated Building and Testing

There is a shortcut for building and running all tests, assuming you've installed the python package
with develop mode. In the root of the `ALTRIOS/` folder, run the `build_and_test.sh` script. In
Windows bash (e.g. git bash), run `sh build_and_test.sh`, or in Linux/Unix, run
`./build_and_test.sh`. This builds all the Rust code, runs Rust tests, builds the Python-exposed
Rust code, and runs the Python tests.

## Manually Building the Python API

Run `maturin develop --release`. Note that not including `--release` will cause a significant
runtime computational performance penalty.

## Testing

### Manually

Whenever updating code, always run `cargo test --release` inside `ALTRIOS/` to ensure that all
tests pass. Also, be sure to rebuild the Python API regularly to ensure that it is up to date.
Python unit tests run with `pytest -v` in the root folder of the git repository.

### With GitHub Actions

Any time anyone pushes to `main` or does any pull request, the [GitHub Actions test
workflows](https://github.com/NREL/altrios/tree/main/.github/workflows) are triggered.

## Releasing

### To PyPI With GitHub Actions

To release the package with GitHub Actions, you can follow these steps:

1. Create a new branch in the format `v<major>.<minor>.<patch>`, for example `v0.2.1`.
1. Update the version number in the `pyproject.toml` file. Commit and push to
   https://github.com/NREL/altrios.
1. Open a pull request into the main branch and make sure all checks pass.
1. Once the pull request is merged into the main branch by a reviewer, create a new GitHub release
   and create a tag that matches the branch name. Once the release is created, a [GitHub
   action](https://github.com/NREL/altrios/blob/686e8c28828cb980cc45567d08091e69b7bee52c/.github/workflows/wheels.yaml#L5)
   will be launched to build the wheels and publish them to PyPI.

### To crates.io

#### altrios-core

If you've updated `altrios-proc-macros`, be sure to publish that crate first and then update the
Cargo.toml dependency for this crate.

To release this crate, you need to be setup as developer for this crate in crates.io. Follow these steps:

1. Increment the version number in
   [altrios-core/Cargo.toml](https://github.com/NREL/altrios/blob/426f50e4ebd0fbf1d7e346aa31604107df8f83fe/altrios-core/Cargo.toml#L8): 
   
   `version = "0.2.1"`.
1. If changes were made in `altrios-proc-macros`, follow [the release process for that
   crate](#altrios-proc-macros) first, and then update the `altrios-proc-macros` dependency version
   to match the new `altrios-proc-macros` version in `altrios-core/Cargo.toml`.
1. Run `git tag ac<major>.<minor>.<patch>`, where `ac<major>.<minor>.<patch>` should look like
   `ac0.1.4`, reflecting whatever the current version is.
1. Push the tag with `git push public ac<major>.<minor>.<patch>`, where `public` is this remote:
   `git@github.com:NREL/altrios.git`.
1. Run `cargo publish --dry-run` to make sure everything checks.
1. Run `cargo publish` to release the update.

In the future, we may incorporate this into GitHub Actions.

#### altrios-proc-macros

To release this crate, you need to be setup as developer for this crate in crates.io. Follow these steps:

1. Increment the version number in
   [Cargo.toml](https://github.com/NREL/altrios/blob/dced44b42c456da88363d03dc43259b039a94e6d/Cargo.toml#L48):
   
   `altrios-proc-macros = { path = "./altrios-core/altrios-proc-macros", version = "0.2.0" }`.
1. Run `git tag apm<major>.<minor>.<patch>`, where `apm<major>.<minor>.<patch>` should look like
   `apm0.1.4`, reflecting whatever the current version is.
1. Push the tag with `git push public apm<major>.<minor>.<patch>`, where `public` is this remote:
   `git@github.com:NREL/altrios.git`.
1. Run `cargo publish --dry-run` to make sure everything checks.
1. Run `cargo publish` to release the update.

In the future, we may incorporate this into GitHub Actions.
