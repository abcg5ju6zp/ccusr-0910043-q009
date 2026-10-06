"""项目内部接口说明。"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import itertools
import json
import os
import re
import typing as t
import warnings
from fnmatch import fnmatch

from jupyter_core.utils import ensure_async, run_sync
from jupyter_events import EventLogger
from nbformat import ValidationError, sign
from nbformat import validate as validate_nb
from nbformat.v4 import new_notebook
from tornado.web import HTTPError, RequestHandler
from traitlets import (
    Any,
    Bool,
    Dict,
    Float,
    Instance,
    Int,
    List,
    TraitError,
    Type,
    Unicode,
    default,
    validate,
)
from traitlets.config.configurable import LoggingConfigurable

from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH, JUPYTER_SERVER_EVENTS_URI
from jupyter_server.transutils import _i18n
from jupyter_server.utils import import_item

from ...files.handlers import FilesHandler
from .checkpoints import AsyncCheckpoints, Checkpoints
from .pagination import (
    CLASSIFY_LIMIT,
    END,
    ListingCursorStore,
    compute_changes,
    empty_changes,
    slice_page,
)

copy_pat = re.compile(r"\-Copy\d*\.")


def _page_payload(session, page, page_size, next_token):
    """Assemble the ``page`` metadata dict of a paginated directory model."""
    return {
        "cursor": next_token,
        "has_more": page["has_more"],
        "page_size": page_size,
        "returned": len(page["content"]),
        "skipped": page["skipped"],
        "sort": session.sort,
        "sort_dir": session.sort_dir,
        "hashes_deferred": page["hashes_deferred"],
        "snapshot": {
            "created": session.created_at,
            "expires_at": session.expires_at(),
            "entries": len(session.names),
            "watermark": session.watermark_display,
        },
        "changes": page["changes"],
    }


class ContentsManager(LoggingConfigurable):
    """项目内部接口说明。"""

    event_schema_id = JUPYTER_SERVER_EVENTS_URI + "/contents_service/v1"
    event_logger = Instance(EventLogger).tag(config=True)

    @default("event_logger")
    def _default_event_logger(self):
        if self.parent and hasattr(self.parent, "event_logger"):
            return self.parent.event_logger
        else:
            # If parent does not have an event logger, create one.
            logger = EventLogger()
            schema_path = DEFAULT_EVENTS_SCHEMA_PATH / "contents_service" / "v1.yaml"
            logger.register_event_schema(schema_path)
            return logger

    def emit(self, data):
        """项目内部接口说明。"""
        self.event_logger.emit(schema_id=self.event_schema_id, data=data)

    root_dir = Unicode("/", config=True)

    preferred_dir = Unicode(
        "",
        config=True,
        help=_i18n(
            "Preferred starting directory to use for notebooks. This is an API path (`/` separated, relative to root dir)"
        ),
    )

    @validate("preferred_dir")
    def _validate_preferred_dir(self, proposal):
        value = proposal["value"].strip("/")
        try:
            import inspect

            if inspect.iscoroutinefunction(self.dir_exists):
                dir_exists = run_sync(self.dir_exists)(value)
            else:
                dir_exists = self.dir_exists(value)
        except HTTPError as e:
            raise TraitError(e.log_message) from e
        if not dir_exists:
            raise TraitError(_i18n("Preferred directory not found: %r") % value)
        if self.parent:
            try:
                if value != self.parent.preferred_dir:
                    self.parent.preferred_dir = os.path.join(self.root_dir, *value.split("/"))
            except TraitError:
                pass
        return value

    allow_hidden = Bool(False, config=True, help="Allow access to hidden files")

    notary = Instance(sign.NotebookNotary)

    @default("notary")
    def _notary_default(self):
        return sign.NotebookNotary(parent=self)

    hide_globs = List(
        Unicode(),
        [
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
            "*~",
        ],
        config=True,
        help="""
        Glob patterns to hide in file and directory listings.
    """,
    )

    untitled_notebook = Unicode(
        _i18n("Untitled"),
        config=True,
        help="The base name used when creating untitled notebooks.",
    )

    untitled_file = Unicode(
        "untitled", config=True, help="The base name used when creating untitled files."
    )

    untitled_directory = Unicode(
        "Untitled Folder",
        config=True,
        help="The base name used when creating untitled directories.",
    )

    pre_save_hook = Any(
        None,
        config=True,
        allow_none=True,
        help="""Python callable or importstring thereof

        To be called on a contents model prior to save.

        This can be used to process the structure,
        such as removing notebook outputs or other side effects that
        should not be saved.

        It will be called as (all arguments passed by keyword)::

            hook(path=path, model=model, contents_manager=self)

        - model: the model to be saved. Includes file contents.
          Modifying this dict will affect the file that is stored.
        - path: the API path of the save destination
        - contents_manager: this ContentsManager instance
        """,
    )

    @validate("pre_save_hook")
    def _validate_pre_save_hook(self, proposal):
        value = proposal["value"]
        if isinstance(value, str):
            value = import_item(self.pre_save_hook)
        if not callable(value):
            msg = "pre_save_hook must be callable"
            raise TraitError(msg)
        if callable(self.pre_save_hook):
            warnings.warn(
                f"Overriding existing pre_save_hook ({self.pre_save_hook.__name__}) with a new one ({value.__name__}).",
                stacklevel=2,
            )
        return value

    post_save_hook = Any(
        None,
        config=True,
        allow_none=True,
        help="""Python callable or importstring thereof

        to be called on the path of a file just saved.

        This can be used to process the file on disk,
        such as converting the notebook to a script or HTML via nbconvert.

        It will be called as (all arguments passed by keyword)::

            hook(os_path=os_path, model=model, contents_manager=instance)

        - path: the filesystem path to the file just written
        - model: the model representing the file
        - contents_manager: this ContentsManager instance
        """,
    )

    @validate("post_save_hook")
    def _validate_post_save_hook(self, proposal):
        value = proposal["value"]
        if isinstance(value, str):
            value = import_item(value)
        if not callable(value):
            msg = "post_save_hook must be callable"
            raise TraitError(msg)
        if callable(self.post_save_hook):
            warnings.warn(
                f"Overriding existing post_save_hook ({self.post_save_hook.__name__}) with a new one ({value.__name__}).",
                stacklevel=2,
            )
        return value

    def run_pre_save_hook(self, model, path, **kwargs):
        """项目内部接口说明。"""
        warnings.warn(
            "run_pre_save_hook is deprecated, use run_pre_save_hooks instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.pre_save_hook:
            try:
                self.log.debug("Running pre-save hook on %s", path)
                self.pre_save_hook(model=model, path=path, contents_manager=self, **kwargs)
            except HTTPError:
                # allow custom HTTPErrors to raise,
                # rejecting the save with a message.
                raise
            except Exception:
                # unhandled errors don't prevent saving,
                # which could cause frustrating data loss
                self.log.error("Pre-save hook failed on %s", path, exc_info=True)

    def run_post_save_hook(self, model, os_path):
        """项目内部接口说明。"""
        warnings.warn(
            "run_post_save_hook is deprecated, use run_post_save_hooks instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.post_save_hook:
            try:
                self.log.debug("Running post-save hook on %s", os_path)
                self.post_save_hook(os_path=os_path, model=model, contents_manager=self)
            except Exception:
                self.log.error("Post-save hook failed o-n %s", os_path, exc_info=True)
                msg = "fUnexpected error while running post hook save: {e}"
                raise HTTPError(500, msg) from None

    _pre_save_hooks: List[t.Any] = List()
    _post_save_hooks: List[t.Any] = List()

    def register_pre_save_hook(self, hook):
        """项目内部接口说明。"""
        if isinstance(hook, str):
            hook = import_item(hook)
        if not callable(hook):
            msg = "hook must be callable"
            raise RuntimeError(msg)
        self._pre_save_hooks.append(hook)

    def register_post_save_hook(self, hook):
        """项目内部接口说明。"""
        if isinstance(hook, str):
            hook = import_item(hook)
        if not callable(hook):
            msg = "hook must be callable"
            raise RuntimeError(msg)
        self._post_save_hooks.append(hook)

    def run_pre_save_hooks(self, model, path, **kwargs):
        """项目内部接口说明。"""
        pre_save_hooks = [self.pre_save_hook] if self.pre_save_hook is not None else []
        pre_save_hooks += self._pre_save_hooks
        for pre_save_hook in pre_save_hooks:
            try:
                self.log.debug("Running pre-save hook on %s", path)
                pre_save_hook(model=model, path=path, contents_manager=self, **kwargs)
            except HTTPError:
                # allow custom HTTPErrors to raise,
                # rejecting the save with a message.
                raise
            except Exception:
                # unhandled errors don't prevent saving,
                # which could cause frustrating data loss
                self.log.error(
                    "Pre-save hook %s failed on %s",
                    pre_save_hook.__name__,
                    path,
                    exc_info=True,
                )

    def run_post_save_hooks(self, model, os_path):
        """项目内部接口说明。"""
        post_save_hooks = [self.post_save_hook] if self.post_save_hook is not None else []
        post_save_hooks += self._post_save_hooks
        for post_save_hook in post_save_hooks:
            try:
                self.log.debug("Running post-save hook on %s", os_path)
                post_save_hook(os_path=os_path, model=model, contents_manager=self)
            except Exception as e:
                self.log.error(
                    "Post-save %s hook failed on %s",
                    post_save_hook.__name__,
                    os_path,
                    exc_info=True,
                )
                raise HTTPError(500, "Unexpected error while running post hook save: %s" % e) from e

    checkpoints_class = Type(Checkpoints, config=True)
    checkpoints = Instance(Checkpoints, config=True)
    checkpoints_kwargs = Dict(config=True)

    @default("checkpoints")
    def _default_checkpoints(self):
        return self.checkpoints_class(**self.checkpoints_kwargs)

    @default("checkpoints_kwargs")
    def _default_checkpoints_kwargs(self):
        return {
            "parent": self,
            "log": self.log,
        }

    files_handler_class = Type(
        FilesHandler,
        klass=RequestHandler,
        allow_none=True,
        config=True,
        help="""handler class to use when serving raw file requests.

        Default is a fallback that talks to the ContentsManager API,
        which may be inefficient, especially for large files.

        Local files-based ContentsManagers can use a StaticFileHandler subclass,
        which will be much more efficient.

        Access to these files should be Authenticated.
        """,
    )

    files_handler_params = Dict(
        config=True,
        help="""Extra parameters to pass to files_handler_class.

        For example, StaticFileHandlers generally expect a `path` argument
        specifying the root directory from which to serve files.
        """,
    )

    def get_extra_handlers(self):
        """项目内部接口说明。"""
        handlers = []
        if self.files_handler_class:
            handlers.append((r"/files/(.*)", self.files_handler_class, self.files_handler_params))
        return handlers

    # ContentsManager API part 1: methods that must be
    # implemented in subclasses.

    def dir_exists(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def is_hidden(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def file_exists(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def exists(self, path):
        """项目内部接口说明。"""
        return self.file_exists(path) or self.dir_exists(path)

    def get(self, path, content=True, type=None, format=None, require_hash=False):
        """项目内部接口说明。"""
        raise NotImplementedError

    def save(self, model, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def delete_file(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    def rename_file(self, old_path, new_path):
        """项目内部接口说明。"""
        raise NotImplementedError

    # ContentsManager API part 2: methods that have usable default
    # implementations, but can be overridden in subclasses.

    def delete(self, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        if not path:
            raise HTTPError(400, "Can't delete root")
        self.delete_file(path)
        self.checkpoints.delete_all_checkpoints(path)
        self.emit(data={"action": "delete", "path": path})

    def rename(self, old_path, new_path):
        """项目内部接口说明。"""
        self.rename_file(old_path, new_path)
        self.checkpoints.rename_all_checkpoints(old_path, new_path)
        self.emit(data={"action": "rename", "path": new_path, "source_path": old_path})

    def update(self, model, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        new_path = model.get("path", path).strip("/")
        if path != new_path:
            self.rename(path, new_path)
        model = self.get(new_path, content=False)
        return model

    def info_string(self):
        """项目内部接口说明。"""
        return "Serving contents"

    def get_kernel_path(self, path, model=None):
        """项目内部接口说明。"""
        return ""

    def increment_filename(self, filename, path="", insert=""):
        """项目内部接口说明。"""
        # Extract the full suffix from the filename (e.g. .tar.gz)
        path = path.strip("/")
        basename, dot, ext = filename.rpartition(".")
        if ext != "ipynb":
            basename, dot, ext = filename.partition(".")

        suffix = dot + ext

        for i in itertools.count():
            insert_i = f"{insert}{i}" if i else ""
            name = f"{basename}{insert_i}{suffix}"
            if not self.exists(f"{path}/{name}"):
                break
        return name

    def validate_notebook_model(self, model, validation_error=None):
        """项目内部接口说明。"""
        try:
            # If we're given a validation_error dictionary, extract the exception
            # from it and raise the exception, else call nbformat's validate method
            # to determine if the notebook is valid.  This 'else' condition may
            # pertain to server extension not using the server's notebook read/write
            # functions.
            if validation_error is not None:
                e = validation_error.get("ValidationError")
                if isinstance(e, ValidationError):
                    raise e
            else:
                validate_nb(model["content"])
        except ValidationError as e:
            model["message"] = "Notebook validation failed: {}:\n{}".format(
                str(e),
                json.dumps(e.instance, indent=1, default=lambda obj: "<UNKNOWN>"),
            )
        return model

    def new_untitled(self, path="", type="", ext=""):
        """项目内部接口说明。"""
        path = path.strip("/")
        if not self.dir_exists(path):
            raise HTTPError(404, "No such directory: %s" % path)

        model = {}
        if type:
            model["type"] = type

        if ext == ".ipynb":
            model.setdefault("type", "notebook")
        else:
            model.setdefault("type", "file")

        insert = ""
        if model["type"] == "directory":
            untitled = self.untitled_directory
            insert = " "
        elif model["type"] == "notebook":
            untitled = self.untitled_notebook
            ext = ".ipynb"
        elif model["type"] == "file":
            untitled = self.untitled_file
        else:
            raise HTTPError(400, "Unexpected model type: %r" % model["type"])

        name = self.increment_filename(untitled + ext, path, insert=insert)
        path = f"{path}/{name}"
        return self.new(model, path)

    def new(self, model=None, path=""):
        """项目内部接口说明。"""
        path = path.strip("/")
        if model is None:
            model = {}

        if path.endswith(".ipynb"):
            model.setdefault("type", "notebook")
        else:
            model.setdefault("type", "file")

        # no content, not a directory, so fill out new-file model
        if "content" not in model and model["type"] != "directory":
            if model["type"] == "notebook":
                model["content"] = new_notebook()
                model["format"] = "json"
            else:
                model["content"] = ""
                model["type"] = "file"
                model["format"] = "text"

        model = self.save(model, path)
        return model

    def copy(self, from_path, to_path=None):
        """项目内部接口说明。"""
        path = from_path.strip("/")

        if to_path is not None:
            to_path = to_path.strip("/")

        if "/" in path:
            from_dir, from_name = path.rsplit("/", 1)
        else:
            from_dir = ""
            from_name = path

        model = self.get(path)
        model.pop("path", None)
        model.pop("name", None)
        if model["type"] == "directory":
            raise HTTPError(400, "Can't copy directories")

        is_destination_specified = to_path is not None
        if not is_destination_specified:
            to_path = from_dir
        if self.dir_exists(to_path):
            name = copy_pat.sub(".", from_name)
            to_name = self.increment_filename(name, to_path, insert="-Copy")
            to_path = f"{to_path}/{to_name}"
        elif is_destination_specified:
            if "/" in to_path:
                to_dir, to_name = to_path.rsplit("/", 1)
                if not self.dir_exists(to_dir):
                    raise HTTPError(404, "No such parent directory: %s to copy file in" % to_dir)
        else:
            raise HTTPError(404, "No such directory: %s" % to_path)

        model = self.save(model, to_path)
        self.emit(data={"action": "copy", "path": to_path, "source_path": from_path})
        return model

    def log_info(self):
        """项目内部接口说明。"""
        self.log.info(self.info_string())

    def trust_notebook(self, path):
        """项目内部接口说明。"""
        model = self.get(path)
        nb = model["content"]
        self.log.warning("Trusting notebook %s", path)
        self.notary.mark_cells(nb, True)
        self.check_and_sign(nb, path)

    def check_and_sign(self, nb, path="", *, _retrying=False):
        """项目内部接口说明。"""
        try:
            if self.notary.check_cells(nb):
                self.notary.sign(nb)
            else:
                self.log.warning("Notebook %s is not trusted", path)
        except Exception:
            if _retrying:
                raise
            self.log.warning(
                "Signature store for notebook %s is corrupted or unavailable; "
                "recreating the store.",
                path,
                exc_info=True,
            )
            # The default implementation uses SQLiteSignatureStore if SQLite3 is available
            # and falls back to MemorySignatureStore if not; SQLiteSignatureStore will
            # attempt to recreate the database if it detects errors during initialization,
            # and fallback to in-memory (`:memory:`) SQLite database if necessary.
            self.notary.store = self.notary.store_factory()
            self.check_and_sign(nb, path, _retrying=True)

    def mark_trusted_cells(self, nb, path=""):
        """项目内部接口说明。"""
        trusted = self.notary.check_signature(nb)
        if not trusted:
            self.log.warning("Notebook %s is not trusted", path)
        self.notary.mark_cells(nb, trusted)

    def should_list(self, name):
        """项目内部接口说明。"""
        return not any(fnmatch(name, glob) for glob in self.hide_globs)

    # Part 2b: paginated directory listings backed by bounded snapshot
    # cursors.  The first page freezes a snapshot (entry names, a fixed
    # sort order, and a watermark of the directory state); later pages are
    # served from that snapshot so concurrent creations/deletions cannot
    # duplicate or skip entries, while visibility rules are re-evaluated
    # on every page so tightened permissions hide entries immediately.

    listing_default_page_size = Int(
        200,
        config=True,
        help="Default page size for paginated directory listings.",
    )

    listing_max_page_size = Int(
        1000,
        config=True,
        help="Maximum accepted page size for paginated directory listings.",
    )

    listing_cursor_ttl = Float(
        300.0,
        config=True,
        help="Idle seconds after which a listing cursor expires.",
    )

    listing_cursor_max_age = Float(
        3600.0,
        config=True,
        help="Absolute lifetime in seconds of a listing cursor, regardless of activity.",
    )

    listing_cursor_max_sessions = Int(
        32,
        config=True,
        help="Maximum number of concurrent listing cursors; least recently used cursors are evicted.",
    )

    listing_snapshot_max_entries = Int(
        100000,
        config=True,
        help="Maximum directory size accepted for a paginated listing; larger directories are rejected with 413.",
    )

    listing_hash_budget_max = Int(
        128,
        config=True,
        help="Maximum number of entry hashes computed per page of a paginated listing.",
    )

    _listing_cursor_store: ListingCursorStore | None = None

    @property
    def listing_cursor_store(self) -> ListingCursorStore:
        """The bounded, process-local store of listing sessions."""
        if self._listing_cursor_store is None:
            self._listing_cursor_store = ListingCursorStore(
                max_sessions=self.listing_cursor_max_sessions,
                ttl=self.listing_cursor_ttl,
                max_age=self.listing_cursor_max_age,
            )
        return self._listing_cursor_store

    def get_page(
        self,
        path,
        page_size=None,
        cursor=None,
        sort=None,
        sort_dir=None,
        require_hash=False,
        hash_budget=None,
    ):
        """Get one page of a directory listing through a bounded snapshot cursor.

        With ``cursor=None`` a new snapshot session is started: entry names,
        the sort order (``sort``/``sort_dir``), and a watermark of the
        directory state are frozen, and the first page is returned together
        with an opaque cursor in ``model["page"]["cursor"]``.  Passing that
        cursor back serves the following pages from the frozen snapshot, so
        concurrent additions/removals neither duplicate nor skip entries;
        the ``page.changes`` summary reports which continuity-relevant
        changes the live directory has seen since the snapshot.

        Visibility (``allow_hidden``, ``hide_globs``, hidden files) is
        re-evaluated on every page, so permissions tightened mid-pagination
        hide entries immediately.  Entry hashes are computed lazily, bounded
        by ``hash_budget`` (itself capped by ``listing_hash_budget_max``);
        entries beyond the budget get ``hash=None`` and the page is flagged
        with ``hashes_deferred``.

        Cursors are bounded (idle TTL, absolute age, max sessions) and
        process-local: expired, evicted, or post-restart cursors raise 410,
        consumed cursors outside the retry window raise 409.  Re-presenting
        a recently consumed cursor replays its page idempotently.
        """
        path = path.strip("/")
        store = self.listing_cursor_store

        if cursor is None:
            sort = sort or "name"
            sort_dir = sort_dir or "asc"
            self._check_sort(sort, sort_dir)
            page_size = self._check_page_size(page_size)
            if not self.dir_exists(path):
                if self.file_exists(path):
                    raise HTTPError(400, "%s is not a directory" % path, reason="bad type")
                raise HTTPError(404, "directory does not exist: %r" % path)
            if not self.allow_hidden and self.is_hidden(path):
                raise HTTPError(404, "directory does not exist: %r" % path)
            keys, names, raw_names, wm_sig, wm_display = self._dir_snapshot(path, sort)
            if len(raw_names) > self.listing_snapshot_max_entries:
                raise HTTPError(
                    413,
                    "directory %r has %d entries, exceeding the listing snapshot "
                    "limit of %d" % (path, len(raw_names), self.listing_snapshot_max_entries),
                )
            session = store.create_session(
                path=path,
                sort=sort,
                sort_dir=sort_dir,
                page_size=page_size,
                keys=keys,
                names=names,
                raw_names=raw_names,
                watermark_signature=wm_sig,
                watermark_display=wm_display,
            )
            position_key = None
            advance = True
            consumed_token = None
            replay_next_token = None
        else:
            session = store.locate(cursor)
            if session.path != path:
                raise HTTPError(
                    400,
                    "Listing cursor belongs to a different directory: %r" % session.path,
                )
            if sort is not None and sort != session.sort:
                raise HTTPError(
                    400,
                    "sort %r does not match the sort order fixed by the first page (%r)"
                    % (sort, session.sort),
                )
            if sort_dir is not None and sort_dir != session.sort_dir:
                raise HTTPError(
                    400,
                    "sort_dir %r does not match the sort order fixed by the first page (%r)"
                    % (sort_dir, session.sort_dir),
                )
            page_size = (
                self._check_page_size(page_size) if page_size is not None else session.page_size
            )
            position_key, advance, replay_next_token = session.classify(cursor)
            consumed_token = cursor if advance else None

        page = self._serve_page(
            session, position_key, page_size, require_hash, hash_budget, is_first=cursor is None
        )

        if advance:
            next_token = session.issue(position_key, consumed_token, page["new_position_key"])
        else:
            next_token = replay_next_token
        session.touch()

        model = self.get(path, content=False)
        model["content"] = page["content"]
        model["format"] = "json"
        model["page"] = _page_payload(session, page, page_size, next_token)
        return model

    def _check_page_size(self, page_size):
        """Validate an explicit page size, or return the configured default."""
        if page_size is None:
            return self.listing_default_page_size
        if not 1 <= page_size <= self.listing_max_page_size:
            raise HTTPError(
                400,
                "page_size must be between 1 and %d" % self.listing_max_page_size,
            )
        return page_size

    def _check_sort(self, sort, sort_dir):
        """Validate the sort order fixed by the first page of a listing."""
        if sort not in ("name", "last_modified"):
            raise HTTPError(400, "sort %r is invalid; must be 'name' or 'last_modified'" % sort)
        if sort_dir not in ("asc", "desc"):
            raise HTTPError(400, "sort_dir %r is invalid; must be 'asc' or 'desc'" % sort_dir)

    def _name_visible(self, name):
        """Cheap name-based visibility check applied to the whole snapshot."""
        return self.should_list(name) and (self.allow_hidden or not name.startswith("."))

    def _serve_page(self, session, position_key, page_size, require_hash, hash_budget, is_first):
        """Build one page of entry models out of the session snapshot."""
        fingerprint = (self.allow_hidden, tuple(self.hide_globs))
        fkeys, fnames = session.filtered_view(fingerprint, self._name_visible)
        page_names, new_position_key, has_more = slice_page(
            fkeys, fnames, position_key, page_size, session.sort_dir
        )

        budget = 0
        if require_hash:
            budget = self.listing_hash_budget_max if hash_budget is None else hash_budget
            budget = max(0, min(budget, self.listing_hash_budget_max))

        content = []
        skipped = 0
        hashed = 0
        hashes_deferred = False
        for name in page_names:
            entry_path = f"{session.path}/{name}"
            try:
                # Re-check visibility live: permissions tightened since the
                # snapshot must hide the entry from this page already.
                if not self.allow_hidden and self.is_hidden(entry_path):
                    skipped += 1
                    continue
                if not self._entry_listable(entry_path, name):
                    skipped += 1
                    continue
                want_hash = require_hash and hashed < budget
                entry_model = self._entry_model(entry_path, want_hash)
            except HTTPError as e:
                if e.status_code in (403, 404):
                    # Vanished, hidden, or unreadable since the snapshot.
                    skipped += 1
                    continue
                raise
            except OSError:
                skipped += 1
                continue
            if want_hash and entry_model.get("hash") is not None:
                hashed += 1
            if (
                require_hash
                and entry_model.get("type") != "directory"
                and entry_model.get("hash") is None
            ):
                hashes_deferred = True
            content.append(entry_model)

        changes = None if is_first else self._session_changes(session)
        return {
            "content": content,
            "skipped": skipped,
            "hashes_deferred": hashes_deferred,
            "new_position_key": new_position_key,
            "has_more": has_more,
            "changes": changes,
        }

    def _session_changes(self, session):
        """Change summary of the live directory against the session watermark."""
        try:
            signature = self._dir_signature(session.path)
            if signature == session.watermark_signature:
                return empty_changes()
            cached = session.changes_cache.get(signature)
            if cached is not None:
                session.changes_cache.move_to_end(signature)
                return cached
            current_names = self._dir_entry_names(session.path)
        except OSError:
            raise HTTPError(404, "directory no longer exists: %r" % session.path) from None
        changes = compute_changes(
            session,
            current_names,
            visible=self._name_visible,
            added_keys=self._added_key_map(session, current_names),
        )
        session.changes_cache[signature] = changes
        while len(session.changes_cache) > 4:
            session.changes_cache.popitem(last=False)
        return changes

    def _added_key_map(self, session, current_names):
        """Sort keys for newly appeared entries, bounded by CLASSIFY_LIMIT."""
        if session.sort == "name":
            return {}
        keys = {}
        added = [n for n in current_names if n not in session.raw_name_set]
        for name in added[:CLASSIFY_LIMIT]:
            try:
                keys[name] = self._entry_sort_key(session.path, name)
            except (HTTPError, OSError):
                continue
        return keys

    # Primitives the paginated flow relies on; the default implementations
    # go through the regular contents API, and file-based managers override
    # them with cheaper direct versions.

    def _dir_snapshot(self, path, sort):
        """Freeze a directory snapshot: (keys, names, raw_names, signature, display)."""
        model = self.get(path, content=True)
        entries = model.get("content") or []
        raw_names = [entry["name"] for entry in entries]
        if sort == "name":
            pairs = sorted((entry["name"], entry["name"]) for entry in entries)
        else:
            pairs = sorted(
                ((entry["last_modified"], entry["name"]), entry["name"]) for entry in entries
            )
        keys = [key for key, _ in pairs]
        names = [name for _, name in pairs]
        signature = (str(model.get("last_modified")), len(entries))
        display = {"last_modified": model.get("last_modified"), "entry_count": len(entries)}
        return keys, names, raw_names, signature, display

    def _dir_signature(self, path):
        """A cheap, hashable signature of the current directory state."""
        model = self.get(path, content=False)
        return (str(model.get("last_modified")),)

    def _dir_entry_names(self, path):
        """The current entry names of the directory."""
        model = self.get(path, content=True)
        return [entry["name"] for entry in model.get("content") or []]

    def _entry_model(self, path, require_hash):
        """The contents model of a single entry."""
        try:
            return self.get(path, content=False, require_hash=require_hash)
        except TypeError:
            # ContentsManager not handling the require_hash argument.
            return self.get(path, content=False)

    def _entry_listable(self, path, name):
        """Whether an entry may appear in a listing (type checks etc.)."""
        return True

    def _entry_sort_key(self, path, name):
        """The sort key of a newly appeared entry, for change classification."""
        model = self.get(f"{path}/{name}", content=False)
        return (model["last_modified"], name)

    # Part 3: Checkpoints API
    def create_checkpoint(self, path):
        """项目内部接口说明。"""
        return self.checkpoints.create_checkpoint(self, path)

    def restore_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        self.checkpoints.restore_checkpoint(self, checkpoint_id, path)

    def list_checkpoints(self, path):
        return self.checkpoints.list_checkpoints(path)

    def delete_checkpoint(self, checkpoint_id, path):
        return self.checkpoints.delete_checkpoint(checkpoint_id, path)


class AsyncContentsManager(ContentsManager):
    """项目内部接口说明。"""

    checkpoints_class = Type(AsyncCheckpoints, config=True)
    checkpoints = Instance(AsyncCheckpoints, config=True)
    checkpoints_kwargs = Dict(config=True)

    @default("checkpoints")
    def _default_checkpoints(self):
        return self.checkpoints_class(**self.checkpoints_kwargs)

    @default("checkpoints_kwargs")
    def _default_checkpoints_kwargs(self):
        return {
            "parent": self,
            "log": self.log,
        }

    # ContentsManager API part 1: methods that must be
    # implemented in subclasses.

    async def dir_exists(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def is_hidden(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def file_exists(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def exists(self, path):
        """项目内部接口说明。"""
        return await ensure_async(self.file_exists(path)) or await ensure_async(
            self.dir_exists(path)
        )

    async def get(self, path, content=True, type=None, format=None, require_hash=False):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def save(self, model, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def delete_file(self, path):
        """项目内部接口说明。"""
        raise NotImplementedError

    async def rename_file(self, old_path, new_path):
        """项目内部接口说明。"""
        raise NotImplementedError

    # ContentsManager API part 2: methods that have usable default
    # implementations, but can be overridden in subclasses.

    async def resolve_path(self, path: str) -> str | None:
        """项目内部接口说明。"""
        return None

    async def delete(self, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        if not path:
            raise HTTPError(400, "Can't delete root")

        await self.delete_file(path)
        await self.checkpoints.delete_all_checkpoints(path)
        self.emit(data={"action": "delete", "path": path})

    async def rename(self, old_path, new_path):
        """项目内部接口说明。"""
        await self.rename_file(old_path, new_path)
        await self.checkpoints.rename_all_checkpoints(old_path, new_path)
        self.emit(data={"action": "rename", "path": new_path, "source_path": old_path})

    async def update(self, model, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        new_path = model.get("path", path).strip("/")
        if path != new_path:
            await self.rename(path, new_path)
        model = await self.get(new_path, content=False)
        return model

    async def increment_filename(self, filename, path="", insert=""):
        """项目内部接口说明。"""
        # Extract the full suffix from the filename (e.g. .tar.gz)
        path = path.strip("/")
        basename, dot, ext = filename.rpartition(".")
        if ext != "ipynb":
            basename, dot, ext = filename.partition(".")

        suffix = dot + ext

        for i in itertools.count():
            insert_i = f"{insert}{i}" if i else ""
            name = f"{basename}{insert_i}{suffix}"
            file_exists = await ensure_async(self.exists(f"{path}/{name}"))
            if not file_exists:
                break
        return name

    async def new_untitled(self, path="", type="", ext=""):
        """项目内部接口说明。"""
        path = path.strip("/")
        dir_exists = await ensure_async(self.dir_exists(path))
        if not dir_exists:
            raise HTTPError(404, "No such directory: %s" % path)

        model = {}
        if type:
            model["type"] = type

        if ext == ".ipynb":
            model.setdefault("type", "notebook")
        else:
            model.setdefault("type", "file")

        insert = ""
        if model["type"] == "directory":
            untitled = self.untitled_directory
            insert = " "
        elif model["type"] == "notebook":
            untitled = self.untitled_notebook
            ext = ".ipynb"
        elif model["type"] == "file":
            untitled = self.untitled_file
        else:
            raise HTTPError(400, "Unexpected model type: %r" % model["type"])

        name = await self.increment_filename(untitled + ext, path, insert=insert)
        path = f"{path}/{name}"
        return await self.new(model, path)

    async def new(self, model=None, path=""):
        """项目内部接口说明。"""
        path = path.strip("/")
        if model is None:
            model = {}

        if path.endswith(".ipynb"):
            model.setdefault("type", "notebook")
        else:
            model.setdefault("type", "file")

        # no content, not a directory, so fill out new-file model
        if "content" not in model and model["type"] != "directory":
            if model["type"] == "notebook":
                model["content"] = new_notebook()
                model["format"] = "json"
            else:
                model["content"] = ""
                model["type"] = "file"
                model["format"] = "text"

        model = await self.save(model, path)
        return model

    async def copy(self, from_path, to_path=None):
        """项目内部接口说明。"""
        path = from_path.strip("/")

        if to_path is not None:
            to_path = to_path.strip("/")

        if "/" in path:
            from_dir, from_name = path.rsplit("/", 1)
        else:
            from_dir = ""
            from_name = path

        model = await self.get(path)
        model.pop("path", None)
        model.pop("name", None)
        if model["type"] == "directory":
            raise HTTPError(400, "Can't copy directories")

        is_destination_specified = to_path is not None
        if not is_destination_specified:
            to_path = from_dir
        if await ensure_async(self.dir_exists(to_path)):
            name = copy_pat.sub(".", from_name)
            to_name = await self.increment_filename(name, to_path, insert="-Copy")
            to_path = f"{to_path}/{to_name}"
        elif is_destination_specified:
            if "/" in to_path:
                to_dir, to_name = to_path.rsplit("/", 1)
                if not await ensure_async(self.dir_exists(to_dir)):
                    raise HTTPError(404, "No such parent directory: %s to copy file in" % to_dir)
        else:
            raise HTTPError(404, "No such directory: %s" % to_path)

        model = await self.save(model, to_path)
        self.emit(data={"action": "copy", "path": to_path, "source_path": from_path})
        return model

    async def trust_notebook(self, path):
        """项目内部接口说明。"""
        model = await self.get(path)
        nb = model["content"]
        self.log.warning("Trusting notebook %s", path)
        self.notary.mark_cells(nb, True)
        self.check_and_sign(nb, path)

    # Part 2b: paginated directory listings; see ContentsManager.get_page
    # for the contract.  This async variant serializes concurrent requests
    # against the same session so a retried page cannot fork the cursor.

    async def get_page(
        self,
        path,
        page_size=None,
        cursor=None,
        sort=None,
        sort_dir=None,
        require_hash=False,
        hash_budget=None,
    ):
        """Get one page of a directory listing through a bounded snapshot cursor."""
        path = path.strip("/")
        store = self.listing_cursor_store

        if cursor is None:
            sort = sort or "name"
            sort_dir = sort_dir or "asc"
            self._check_sort(sort, sort_dir)
            page_size = self._check_page_size(page_size)
            if not await ensure_async(self.dir_exists(path)):
                if await ensure_async(self.file_exists(path)):
                    raise HTTPError(400, "%s is not a directory" % path, reason="bad type")
                raise HTTPError(404, "directory does not exist: %r" % path)
            if not self.allow_hidden and await ensure_async(self.is_hidden(path)):
                raise HTTPError(404, "directory does not exist: %r" % path)
            keys, names, raw_names, wm_sig, wm_display = await self._dir_snapshot(path, sort)
            if len(raw_names) > self.listing_snapshot_max_entries:
                raise HTTPError(
                    413,
                    "directory %r has %d entries, exceeding the listing snapshot "
                    "limit of %d" % (path, len(raw_names), self.listing_snapshot_max_entries),
                )
            session = store.create_session(
                path=path,
                sort=sort,
                sort_dir=sort_dir,
                page_size=page_size,
                keys=keys,
                names=names,
                raw_names=raw_names,
                watermark_signature=wm_sig,
                watermark_display=wm_display,
            )
        else:
            session = store.locate(cursor)
            if session.path != path:
                raise HTTPError(
                    400,
                    "Listing cursor belongs to a different directory: %r" % session.path,
                )
            if sort is not None and sort != session.sort:
                raise HTTPError(
                    400,
                    "sort %r does not match the sort order fixed by the first page (%r)"
                    % (sort, session.sort),
                )
            if sort_dir is not None and sort_dir != session.sort_dir:
                raise HTTPError(
                    400,
                    "sort_dir %r does not match the sort order fixed by the first page (%r)"
                    % (sort_dir, session.sort_dir),
                )
            page_size = (
                self._check_page_size(page_size) if page_size is not None else session.page_size
            )

        async with session.lock:
            if cursor is None:
                position_key, advance, consumed_token, replay_next_token = None, True, None, None
            else:
                position_key, advance, replay_next_token = session.classify(cursor)
                consumed_token = cursor if advance else None
            page = await self._serve_page(
                session, position_key, page_size, require_hash, hash_budget, is_first=cursor is None
            )
            if advance:
                next_token = session.issue(position_key, consumed_token, page["new_position_key"])
            else:
                next_token = replay_next_token
            session.touch()

        model = await self.get(path, content=False)
        model["content"] = page["content"]
        model["format"] = "json"
        model["page"] = _page_payload(session, page, page_size, next_token)
        return model

    async def _serve_page(
        self, session, position_key, page_size, require_hash, hash_budget, is_first
    ):
        """Build one page of entry models out of the session snapshot."""
        fingerprint = (self.allow_hidden, tuple(self.hide_globs))
        fkeys, fnames = session.filtered_view(fingerprint, self._name_visible)
        page_names, new_position_key, has_more = slice_page(
            fkeys, fnames, position_key, page_size, session.sort_dir
        )

        budget = 0
        if require_hash:
            budget = self.listing_hash_budget_max if hash_budget is None else hash_budget
            budget = max(0, min(budget, self.listing_hash_budget_max))

        content = []
        skipped = 0
        hashed = 0
        hashes_deferred = False
        for name in page_names:
            entry_path = f"{session.path}/{name}"
            try:
                # Re-check visibility live: permissions tightened since the
                # snapshot must hide the entry from this page already.
                if not self.allow_hidden and await ensure_async(self.is_hidden(entry_path)):
                    skipped += 1
                    continue
                if not await self._entry_listable(entry_path, name):
                    skipped += 1
                    continue
                want_hash = require_hash and hashed < budget
                entry_model = await self._entry_model(entry_path, want_hash)
            except HTTPError as e:
                if e.status_code in (403, 404):
                    # Vanished, hidden, or unreadable since the snapshot.
                    skipped += 1
                    continue
                raise
            except OSError:
                skipped += 1
                continue
            if want_hash and entry_model.get("hash") is not None:
                hashed += 1
            if (
                require_hash
                and entry_model.get("type") != "directory"
                and entry_model.get("hash") is None
            ):
                hashes_deferred = True
            content.append(entry_model)

        changes = None if is_first else await self._session_changes(session)
        return {
            "content": content,
            "skipped": skipped,
            "hashes_deferred": hashes_deferred,
            "new_position_key": new_position_key,
            "has_more": has_more,
            "changes": changes,
        }

    async def _session_changes(self, session):
        """Change summary of the live directory against the session watermark."""
        try:
            signature = await self._dir_signature(session.path)
            if signature == session.watermark_signature:
                return empty_changes()
            cached = session.changes_cache.get(signature)
            if cached is not None:
                session.changes_cache.move_to_end(signature)
                return cached
            current_names = await self._dir_entry_names(session.path)
        except OSError:
            raise HTTPError(404, "directory no longer exists: %r" % session.path) from None
        changes = compute_changes(
            session,
            current_names,
            visible=self._name_visible,
            added_keys=await self._added_key_map(session, current_names),
        )
        session.changes_cache[signature] = changes
        while len(session.changes_cache) > 4:
            session.changes_cache.popitem(last=False)
        return changes

    async def _added_key_map(self, session, current_names):
        """Sort keys for newly appeared entries, bounded by CLASSIFY_LIMIT."""
        if session.sort == "name":
            return {}
        keys = {}
        added = [n for n in current_names if n not in session.raw_name_set]
        for name in added[:CLASSIFY_LIMIT]:
            try:
                keys[name] = await self._entry_sort_key(session.path, name)
            except (HTTPError, OSError):
                continue
        return keys

    # Primitives the paginated flow relies on; see ContentsManager for the
    # synchronous default implementations.

    async def _dir_snapshot(self, path, sort):
        """Freeze a directory snapshot: (keys, names, raw_names, signature, display)."""
        model = await self.get(path, content=True)
        entries = model.get("content") or []
        raw_names = [entry["name"] for entry in entries]
        if sort == "name":
            pairs = sorted((entry["name"], entry["name"]) for entry in entries)
        else:
            pairs = sorted(
                ((entry["last_modified"], entry["name"]), entry["name"]) for entry in entries
            )
        keys = [key for key, _ in pairs]
        names = [name for _, name in pairs]
        signature = (str(model.get("last_modified")), len(entries))
        display = {"last_modified": model.get("last_modified"), "entry_count": len(entries)}
        return keys, names, raw_names, signature, display

    async def _dir_signature(self, path):
        """A cheap, hashable signature of the current directory state."""
        model = await self.get(path, content=False)
        return (str(model.get("last_modified")),)

    async def _dir_entry_names(self, path):
        """The current entry names of the directory."""
        model = await self.get(path, content=True)
        return [entry["name"] for entry in model.get("content") or []]

    async def _entry_model(self, path, require_hash):
        """The contents model of a single entry."""
        try:
            return await self.get(path, content=False, require_hash=require_hash)
        except TypeError:
            # ContentsManager not handling the require_hash argument.
            return await self.get(path, content=False)

    async def _entry_listable(self, path, name):
        """Whether an entry may appear in a listing (type checks etc.)."""
        return True

    async def _entry_sort_key(self, path, name):
        """The sort key of a newly appeared entry, for change classification."""
        model = await self.get(f"{path}/{name}", content=False)
        return (model["last_modified"], name)

    # Part 3: Checkpoints API
    async def create_checkpoint(self, path):
        """项目内部接口说明。"""
        return await self.checkpoints.create_checkpoint(self, path)

    async def restore_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        await self.checkpoints.restore_checkpoint(self, checkpoint_id, path)

    async def list_checkpoints(self, path):
        """项目内部接口说明。"""
        return await self.checkpoints.list_checkpoints(path)

    async def delete_checkpoint(self, checkpoint_id, path):
        """项目内部接口说明。"""
        return await self.checkpoints.delete_checkpoint(checkpoint_id, path)
