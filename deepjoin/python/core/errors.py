class NoSamplesError(RuntimeError):
    """Raised when the data cannot be sampled correctly"""

    pass


class IsosurfaceExtractionError(RuntimeError):
    """Raised when the network cannot extract an isosurface"""

    pass


class DecoderNotLoadedError(RuntimeError):
    """Raised when the user tries to generate a mesh without a decoder"""

    pass


class PathAccessError(RuntimeError):
    """Raised when the user accesses a path that they shouldn't"""

    pass


class MeshEmptyError(RuntimeError):
    """Raised when a mesh is empty"""

    pass


class MeshContainsError(RuntimeError):
    """Raised when check_contains fails"""

    pass
