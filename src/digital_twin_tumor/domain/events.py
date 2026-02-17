"""Lightweight event bus for decoupled intra-process communication.

Provides a simple publish/subscribe mechanism so that domain modules can react
to events (e.g. ``lesion_detected``, ``recist_classified``) without hard
dependencies on each other.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(UTC).replace(tzinfo=None)


# Canonical pipeline event types (ADR-001)
STUDY_INGESTED = "study.ingested"
VOLUME_PREPROCESSED = "volume.preprocessed"
LESION_MEASURED = "lesion.measured"
IDENTITY_CONFIRMED = "lesion.identity_confirmed"
TWIN_FITTED = "twin.fitted"
NARRATIVE_GENERATED = "narrative.generated"
MEASUREMENT_OVERRIDDEN = "measurement.overridden"
RECIST_CLASSIFIED = "recist.classified"
SIMULATION_COMPLETED = "simulation.completed"
AUDIT_LOGGED = "audit.logged"


@dataclass(frozen=True)
class Event:
    """An immutable domain event.

    Attributes
    ----------
    type:
        A string identifier for the event category, e.g. ``"lesion_detected"``.
    payload:
        Arbitrary data associated with the event.
    timestamp:
        UTC time at which the event was created.
    """

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_now)


# Type alias for subscriber callbacks.
EventHandler = Callable[[Event], None]


class EventBus:
    """A synchronous, in-process publish/subscribe event bus.

    Example
    -------
    >>> bus = EventBus()
    >>> log: list[Event] = []
    >>> bus.subscribe("scan_loaded", log.append)
    >>> bus.publish(Event(type="scan_loaded", payload={"path": "/data/ct"}))
    >>> len(log)
    1
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._lock = threading.Lock()

    # -- public API --------------------------------------------------------

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register *handler* to be called when *event_type* is published.

        Parameters
        ----------
        event_type:
            The event category string to listen for.
        handler:
            A callable that accepts a single :class:`Event` argument.
        """
        with self._lock:
            self._handlers[event_type].append(handler)

    def publish(
        self,
        event: Event | str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Dispatch an event to all registered handlers for its type.

        Accepts either a pre-built :class:`Event` object **or** an event-type
        string with an optional payload dict (convenience form).

        Handlers are invoked synchronously in registration order.  If a handler
        raises an exception it is propagated immediately and subsequent handlers
        are **not** called.

        Parameters
        ----------
        event:
            An :class:`Event` instance, or a string event-type name.
        payload:
            Optional payload dict (used only when *event* is a string).
        """
        if isinstance(event, str):
            event = Event(type=event, payload=payload or {})
        with self._lock:
            handlers = list(self._handlers.get(event.type, []))
        for handler in handlers:
            handler(event)

    def clear(self) -> None:
        """Remove all registered handlers."""
        with self._lock:
            self._handlers.clear()

    # -- introspection -----------------------------------------------------

    def handler_count(self, event_type: str) -> int:
        """Return the number of handlers registered for *event_type*."""
        return len(self._handlers.get(event_type, []))

    @property
    def event_types(self) -> list[str]:
        """Return a sorted list of event types with at least one handler."""
        return sorted(k for k, v in self._handlers.items() if v)
