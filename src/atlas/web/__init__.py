"""Private HTTP application for the local ATLAS paper-trading system."""

def create_app(*args, **kwargs):
    """Import the application factory only when an HTTP server is requested."""
    from atlas.web.app import create_app as factory

    return factory(*args, **kwargs)


__all__ = ["create_app"]
