"""Allow ``python -m polyclustx`` wherever the console command is unavailable."""

from .command import main

if __name__ == "__main__":
    main()
