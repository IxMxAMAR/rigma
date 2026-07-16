"""Enable `python -m rigma` (used by `rigma up --detach` to respawn)."""
from .cli import app

if __name__ == "__main__":
    app()
