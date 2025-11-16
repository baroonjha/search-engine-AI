# Search Engine

Small personal project. Add a short description here about the repository purpose, usage and structure.

## Quick start

- Ensure you have Python installed (recommended via Anaconda/Miniconda).
- (Optional) Create and activate a conda environment:

```powershell
conda create -n search-env python=3.11 -y
conda activate search-env
```

- Install requirements if any (create `requirements.txt` as needed):

```powershell
pip install -r requirements.txt
```

## GitHub

To create a remote repository and push your local repo to GitHub, you can use the `gh` CLI or create the repo on the GitHub website and then add the remote manually.

Using GitHub CLI (recommended if installed and authenticated):

```powershell
# create a new public repo from current folder, add remote and push
gh repo create <OWNER>/<REPO> --public --source=. --remote=origin --push
```

Manual method via GitHub website:

1. On GitHub, create a new repository (do not initialize with README).
2. In your local repo run:

```powershell
git branch -M main
git remote add origin https://github.com/<OWNER>/<REPO>.git
git push -u origin main
```

Replace `<OWNER>` and `<REPO>` with your GitHub username/organization and repository name.

## Notes

- If `git commit` fails due to missing user info, set it with:

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

- If you want me to create the GitHub repository for you using the `gh` CLI, tell me and confirm you have `gh` installed and authenticated; I can run the `gh repo create` command.
