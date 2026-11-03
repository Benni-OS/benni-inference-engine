"""BIE CLI — Legacy bridge and Typer entrypoint delegating to bie.serve.__main__"""
import sys

from bie.serve.__main__ import main


def app():
    """CLI entrypoint."""
    sys.exit(main())

if __name__ == "__main__":
    app()
