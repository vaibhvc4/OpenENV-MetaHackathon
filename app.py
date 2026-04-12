"""Root app entry point — re-exports from server/app.py."""

from server.app import app, main  # noqa: F401

if __name__ == "__main__":
    main()
