from pathlib import Path


GITIGNORE = Path(".gitignore")

CONTENT = """# Python
.venv/
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/

# Editors / OS
.idea/
.vscode/
.DS_Store
Thumbs.db

# Root data directory: never commit raw or derived microdata
/data/

# Notebooks may contain outputs with microdata; clean before publishing
/notebooks/
*.ipynb_checkpoints/

# Survey/binary data files anywhere
*.sav
*.dta
*.por
*.xpt
*.parquet
*.feather
*.pkl
*.pickle

# Temporary terminal logs
/sweep_out.txt
/grid_out.txt
/girls_out.txt
*.backup

# Local secrets
.env
.env.*
"""


def main():
    GITIGNORE.write_text(CONTENT, encoding="utf-8")
    print("Safe .gitignore written.")
    print(f"Path: {GITIGNORE.resolve()}")
    print(f"Lines: {len(CONTENT.strip().splitlines())}")


if __name__ == "__main__":
    main()