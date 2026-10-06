"""Bounded snapshot cursors for paginated directory listings.

This module implements the server-side state machine behind paginated
``GET /api/contents/<dir>`` requests:

* The first page freezes a *snapshot* of the directory: the entry names, a
  fixed sort order, and a *watermark* (directory mtime/ctime and entry
  count) that later pages compare against.
* Pages are served out of the frozen snapshot with keyset (cursor)
  semantics, so concurrent creations and deletions cannot produce
  duplicate or skipped entries within a session.
* Every page re-evaluates visibility (``allow_hidden``/``hide_globs`` and
  friends) against the *current* configuration, so permissions that
  tighten mid-pagination hide entries immediately.
* Cursors are opaque tokens backed by a bounded, process-local store
  (idle TTL, absolute max age, max session count).  Expired, evicted, or
  post-restart tokens fail with an explicit 410; tokens that were already
  consumed and fell out of the retry window fail with an explicit 409.
* Re-presenting a recently consumed token replays the page it served
  (idempotent same-page retry) and hands back the same successor cursor.

The store is deliberately process-local: snapshots trade durability for
bounded memory, and clients are told to restart the listing when a cursor
is no longer honored.
"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import asyncio
import bisect
import time
import typing as t
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from tornado.web import HTTPError

#: Version prefix for opaque cursor tokens: ``dsc1.<session-id>.<serial>``.
CURSOR_PREFIX = "dsc1"

#: Sentinel marking an exhausted listing frontier (no more entries).
END: t.Any = object()

#: How many added/removed names a change summary reports before truncating.
CHANGES_REPORT_LIMIT = 100

#: How many recent consumed tokens can be replayed per session.
HISTORY_SIZE = 16

#: How many filtered views / change summaries are cached per session.
_VIEW_CACHE_SIZE = 4

#: Upper bound on entries stat-ed while classifying added entries for
#: continuity; beyond this the session conservatively reports broken
#: continuity instead of doing unbounded work.
CLASSIFY_LIMIT = 256


def utcnow() -> datetime:
    """Return the current time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def empty_changes() -> dict[str, t.Any]:
    """A change summary describing a directory that matches its watermark."""
    return {
        "continuity_broken": False,
        "added": [],
        "added_count": 0,
        "removed": [],
        "removed_count": 0,
        "truncated": False,
    }


def _in_served_region(key: t.Any, frontier_key: t.Any, sort_dir: str) -> bool:
    """Whether ``key`` falls in the already-served region of a session.

    The frontier key is the first un-served entry, so the served region is
    everything ordered strictly before it in the session's sort direction.
    """
    if sort_dir == "asc":
        return bool(key < frontier_key)
    return bool(key > frontier_key)


class ListingSession:
    """Server-side state for one bounded snapshot listing session.

    Parameters
    ----------
    keys, names:
        The snapshot entries in canonical *ascending* sort order, as two
        parallel lists.  Keys are unique: entry names for the ``name``
        sort, ``(sort_key, name)`` tuples otherwise.
    raw_names:
        The unfiltered names observed when the snapshot was taken, used as
        the baseline for change detection.
    watermark_signature:
        An opaque, hashable value identifying the directory state at
        snapshot time (e.g. ``(st_mtime_ns, st_ctime_ns)``).
    watermark_display:
        JSON-serializable watermark details exposed to clients.
    """

    def __init__(
        self,
        *,
        session_id: str,
        path: str,
        sort: str,
        sort_dir: str,
        page_size: int,
        keys: list[t.Any],
        names: list[str],
        raw_names: list[str],
        watermark_signature: t.Hashable,
        watermark_display: dict[str, t.Any],
        ttl: float,
        max_age: float,
        history_size: int = HISTORY_SIZE,
    ):
        self.session_id = session_id
        self.path = path
        self.sort = sort
        self.sort_dir = sort_dir
        self.page_size = page_size
        self.keys = keys
        self.names = names
        self.raw_names = raw_names
        self.raw_name_set = frozenset(raw_names)
        self.watermark_signature = watermark_signature
        self.watermark_display = watermark_display
        self.ttl = ttl
        self.max_age = max_age
        self.history_size = history_size

        self.created_at = utcnow()
        self._created = time.monotonic()
        self._last_used = self._created

        # name -> sort key lookup for entries removed since the snapshot
        # (only needed when the key is not the name itself).
        self.key_by_name: dict[str, t.Any] | None = (
            None if sort == "name" else dict(zip(names, keys, strict=True))
        )

        # Frontier: sort key of the first un-served entry.  None means the
        # session starts at the beginning; END means it is exhausted.
        self.frontier_key: t.Any = None
        self.frontier_token: str | None = None
        # Recently consumed tokens -> (position served from, successor token),
        # enabling idempotent same-page retries.
        self.history: OrderedDict[str, tuple[t.Any, str | None]] = OrderedDict()
        # Visibility fingerprint -> filtered (keys, names) view.
        self.filtered: OrderedDict[t.Any, tuple[list[t.Any], list[str]]] = OrderedDict()
        # Live directory signature -> computed change summary.
        self.changes_cache: OrderedDict[t.Hashable, dict[str, t.Any]] = OrderedDict()
        # Serializes concurrent requests against this session.
        self.lock = asyncio.Lock()
        self._serial = 0

    def expired(self, now: float | None = None) -> bool:
        """Whether the session has outlived its idle TTL or absolute age."""
        now = time.monotonic() if now is None else now
        return (now - self._last_used) > self.ttl or (now - self._created) > self.max_age

    def touch(self) -> None:
        """Refresh the idle expiry clock."""
        self._last_used = time.monotonic()

    def expires_at(self) -> datetime:
        """Absolute wall-clock time after which the cursor is dropped."""
        return self.created_at + timedelta(seconds=self.max_age)

    def filtered_view(
        self, fingerprint: t.Any, is_visible: t.Callable[[str], bool]
    ) -> tuple[list[t.Any], list[str]]:
        """Snapshot entries passing the *current* visibility rules.

        Views are cached per fingerprint so the common case (unchanged
        configuration) costs nothing, while a tightened configuration
        takes effect on the very next page.
        """
        view = self.filtered.get(fingerprint)
        if view is None:
            pairs = [(k, n) for k, n in zip(self.keys, self.names, strict=True) if is_visible(n)]
            view = ([k for k, _ in pairs], [n for _, n in pairs])
            self.filtered[fingerprint] = view
            while len(self.filtered) > _VIEW_CACHE_SIZE:
                self.filtered.popitem(last=False)
        else:
            self.filtered.move_to_end(fingerprint)
        return view

    def classify(self, token: str) -> tuple[t.Any, bool, str | None]:
        """Classify a presented cursor token for this session.

        Returns ``(position_key, advance, replay_next_token)``:

        * ``advance=True``: the token is the live frontier token; the page
          is served from the frontier and the frontier advances.
        * ``advance=False``: the token was recently consumed; the page is
          replayed from its recorded position and ``replay_next_token``
          (the successor issued back then) is returned unchanged.

        Raises 409 for tokens that are well-formed but no longer honored.
        """
        if token == self.frontier_token:
            return self.frontier_key, True, None
        entry = self.history.get(token)
        if entry is not None:
            position_key, next_token = entry
            return position_key, False, next_token
        raise HTTPError(
            409,
            "Listing cursor was already consumed and fell out of the retry "
            "window; resume with the most recently issued cursor.",
        )

    def issue(
        self, served_position: t.Any, consumed_token: str | None, new_frontier_key: t.Any
    ) -> str | None:
        """Advance the frontier after serving a page; return the next token.

        Returns None when the listing is exhausted.  The consumed token is
        recorded so it can be replayed idempotently within the retry window.
        """
        new_token: str | None = None
        if new_frontier_key is not END:
            self._serial += 1
            new_token = f"{CURSOR_PREFIX}.{self.session_id}.{self._serial}"
        if consumed_token is not None:
            self.history[consumed_token] = (served_position, new_token)
            self.history.move_to_end(consumed_token)
            while len(self.history) > self.history_size:
                self.history.popitem(last=False)
        self.frontier_key = new_frontier_key
        self.frontier_token = new_token
        return new_token


class ListingCursorStore:
    """A bounded, process-local store of listing sessions.

    Bounds: ``max_sessions`` concurrent sessions (LRU-evicted), an idle
    ``ttl``, and an absolute ``max_age`` per session.  Cursors do not
    survive a server restart; unknown or expired cursors raise 410.
    """

    def __init__(
        self,
        *,
        max_sessions: int,
        ttl: float,
        max_age: float,
        history_size: int = HISTORY_SIZE,
    ):
        self.max_sessions = max_sessions
        self.ttl = ttl
        self.max_age = max_age
        self.history_size = history_size
        self._sessions: OrderedDict[str, ListingSession] = OrderedDict()

    def create_session(
        self,
        *,
        path: str,
        sort: str,
        sort_dir: str,
        page_size: int,
        keys: list[t.Any],
        names: list[str],
        raw_names: list[str],
        watermark_signature: t.Hashable,
        watermark_display: dict[str, t.Any],
    ) -> ListingSession:
        """Create and register a new session; evicts expired/LRU sessions."""
        self._sweep()
        session = ListingSession(
            session_id=uuid.uuid4().hex,
            path=path,
            sort=sort,
            sort_dir=sort_dir,
            page_size=page_size,
            keys=keys,
            names=names,
            raw_names=raw_names,
            watermark_signature=watermark_signature,
            watermark_display=watermark_display,
            ttl=self.ttl,
            max_age=self.max_age,
            history_size=self.history_size,
        )
        self._sessions[session.session_id] = session
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
        return session

    def locate(self, token: str) -> ListingSession:
        """Resolve a token to its session, or raise 400/410.

        400: the token is not a well-formed cursor.
        410: the cursor is expired, evicted, or from a previous process.
        """
        parts = token.split(".")
        if len(parts) != 3 or parts[0] != CURSOR_PREFIX or not parts[2].isdigit():
            raise HTTPError(400, "Malformed listing cursor: %r" % token)
        session = self._sessions.get(parts[1])
        if session is None or session.expired():
            self._sessions.pop(parts[1], None)
            raise HTTPError(
                410,
                "Listing cursor expired or unknown. Cursors are bounded and do "
                "not survive a server restart; start a new listing without a "
                "cursor.",
            )
        self._sessions.move_to_end(session.session_id)
        session.touch()
        return session

    def _sweep(self) -> None:
        now = time.monotonic()
        for sid in [sid for sid, s in self._sessions.items() if s.expired(now)]:
            del self._sessions[sid]


def slice_page(
    keys: list[t.Any],
    names: list[str],
    position_key: t.Any,
    page_size: int,
    sort_dir: str,
) -> tuple[list[str], t.Any, bool]:
    """Slice one page out of a (filtered) snapshot view.

    ``position_key`` is the sort key of the first un-served entry (None for
    the beginning of the listing, END for an exhausted one).  Because the
    snapshot is frozen, positions are stable: entries hidden in the
    meantime simply drop out of the view without shifting anything else.

    Returns ``(page_names, new_position_key, has_more)``.
    """
    if position_key is END:
        return [], END, False
    if sort_dir == "asc":
        start = 0 if position_key is None else bisect.bisect_left(keys, position_key)
        stop = start + page_size
        page = names[start:stop]
        if stop < len(names):
            return page, keys[stop], True
        return page, END, False
    # Descending order is served from the top of the ascending view.
    stop = len(keys) if position_key is None else bisect.bisect_right(keys, position_key)
    start = max(0, stop - page_size)
    page = names[start:stop]
    page.reverse()
    if start > 0:
        return page, keys[start - 1], True
    return page, END, False


def compute_changes(
    session: ListingSession,
    current_names: list[str],
    *,
    visible: t.Callable[[str], bool],
    added_keys: dict[str, t.Any] | None = None,
    report_limit: int = CHANGES_REPORT_LIMIT,
) -> dict[str, t.Any]:
    """Diff the live directory against the session's snapshot watermark.

    Only names passing the *current* visibility rules are reported, so the
    summary never leaks entries the caller is not allowed to see.

    ``continuity_broken`` is True when an entry was added to or removed
    from the already-served region of the listing, meaning the pages the
    client has seen no longer form a consistent prefix of the live
    directory.

    For sorts whose key is not the entry name, ``added_keys`` maps newly
    appeared names to their sort keys (pre-computed by the caller, bounded
    by CLASSIFY_LIMIT).  An added entry without a classified key makes the
    summary conservatively report broken continuity.
    """
    current_set = set(current_names)
    added = sorted(n for n in current_set - session.raw_name_set if visible(n))
    removed = sorted(n for n in session.raw_name_set - current_set if visible(n))

    frontier = session.frontier_key
    if frontier is END:
        broken = bool(added or removed)
    elif frontier is None:
        # Nothing has been served yet; any state is still fully consistent.
        broken = False
    else:
        broken = False
        key_by_name = session.key_by_name
        for name in removed:
            key = name if key_by_name is None else key_by_name.get(name)
            if key is not None and _in_served_region(key, frontier, session.sort_dir):
                broken = True
                break
        if not broken and added:
            if session.sort == "name":
                broken = any(_in_served_region(n, frontier, session.sort_dir) for n in added)
            else:
                keys = added_keys or {}
                if any(n not in keys for n in added):
                    # At least one added entry could not be classified
                    # within budget; report conservatively.
                    broken = True
                else:
                    broken = any(
                        _in_served_region(keys[n], frontier, session.sort_dir) for n in added
                    )

    return {
        "continuity_broken": broken,
        "added": added[:report_limit],
        "added_count": len(added),
        "removed": removed[:report_limit],
        "removed_count": len(removed),
        "truncated": len(added) > report_limit or len(removed) > report_limit,
    }
