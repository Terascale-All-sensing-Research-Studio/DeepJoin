class MeshNotClosedError(RuntimeError):
    """Raised when input mesh is not closed 2-manifold."""

    pass


class MeshSizeError(RuntimeError):
    """Raised when input mesh has too many vertices or faces."""

    pass


class MeshNotFoundError(RuntimeError):
    """Raised when a mesh file is not found."""

    pass


class MeshBreakMaxRetriesError(RuntimeError):
    """Raised when a mesh break fails because of too many retries."""

    pass


class MeshEmptyError(RuntimeError):
    """Raised when a mesh contains no vertices."""

    pass


class NoDisplayError(RuntimeError):
    """Raised when display cannot be found."""

    pass


class MeshWaterproofError(RuntimeError):
    """Raised when waterproofing fails."""

    pass


class SplineFitError(RuntimeError):
    """Raised when spline fit has very high error."""

    pass
