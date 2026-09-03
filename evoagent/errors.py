"""Control-flow exceptions shared by the agent loop, the pipeline and serving.

These live outside every feature package so that ``agents``, ``review``,
``tools`` and ``serving`` can raise and catch the same types without importing
each other.
"""


class BudgetExceeded(RuntimeError):
    """A step, token or wall-clock budget was exhausted."""


class TaskCancelled(RuntimeError):
    """The owning task requested cancellation."""


class ToolProtocolError(RuntimeError):
    """A tool request does not match the registered tool contract."""


class InvalidTransition(RuntimeError):
    """A task was asked to move to a state its lifecycle does not allow."""
