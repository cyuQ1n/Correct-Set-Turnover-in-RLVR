# Contributing to ReMind

Everyone is welcome to contribute, and we value everybody's contribution. Code contributions are not the only way to help the community. Answering questions, helping others, and improving the documentation are also immensely valuable.

It also helps us if you spread the word! Reference the library in blog posts about the awesome projects it made possible, or simply star the repository to say thank you.

However you choose to contribute, please be mindful and respect our [code of conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

There are several ways you can contribute to ReMind:

* Fix outstanding issues with the existing code.
* Submit issues related to bugs or desired new features.
* Contribute to the examples or to the documentation.

### Style guide

This project follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), check it for details. We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

### Create a Pull Request

1. Fork the repository by clicking on the Fork button on the repository's page. This creates a copy of the code under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

```bash
git clone git@github.com:<your-username>/Correct-Set-Turnover-in-RLVR.git
cd Correct-Set-Turnover-in-RLVR
git remote add upstream https://github.com/cyuQ1n/Correct-Set-Turnover-in-RLVR.git
```

3. Create a new branch to hold your development changes:

```bash
git checkout -b dev_your_branch
```

4. Set up a development environment by running the following command in a virtual environment:

```bash
pip install -e ".[dev]"
```

5. Check code before commit:

```bash
make style && make quality
```

6. Submit changes:

```bash
git add .
git commit -m "commit message"
git fetch upstream
git rebase upstream/main
git push -u origin dev_your_branch
```

7. Create a pull request from your branch.
