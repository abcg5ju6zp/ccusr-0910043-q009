"""项目内部接口说明。"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import asyncio
import errno
import math
import mimetypes
import os
import platform
import shutil
import stat
import subprocess
import sys
import typing as t
import warnings
from datetime import datetime
from pathlib import Path

import nbformat
from anyio.to_thread import run_sync
from jupyter_core.paths import exists, is_file_hidden, is_hidden
from send2trash import send2trash
from tornado import web
from traitlets import Bool, Int, TraitError, Unicode, default, validate

from jupyter_server import _tz as tz
from jupyter_server.base.handlers import AuthenticatedFileHandler
from jupyter_server.transutils import _i18n
from jupyter_server.utils import to_api_path

from .filecheckpoints import AsyncFileCheckpoints, FileCheckpoints
from .fileio import AsyncFileManagerMixin, FileManagerMixin
from .listing import ListingCursorStore, build_pagination_payload
from .manager import AsyncContentsManager, ContentsManager, copy_pat

try:
    from os.path import samefile
except ImportError:
    # windows
    from jupyter_server.utils import samefile_simple as samefile  # type:ignore[assignment]

_script_exporter = None


def _get_created_timestamp(info: os.stat_result) -> float:
    """项目内部接口说明。"""
    birthtime = getattr(info, "st_birthtime", None)
    # Validate: must be numeric, non-negative, and finite.
    # Some FUSE/network filesystems may return None or non-numeric values.
    # Note: birthtime >= 0 rejects pre-1970 dates as these typically indicate
    # invalid or uninitialized values rather than legitimate historical dates.
    if isinstance(birthtime, (int, float)) and birthtime >= 0 and math.isfinite(birthtime):
        return birthtime
    # Fallback to st_ctime; validation happens in _base_model() during datetime conversion
    # where OverflowError and other conversion errors are caught and handled
    return info.st_ctime


class FileContentsManager(FileManagerMixin, ContentsManager):
    """项目内部接口说明。"""

    root_dir = Unicode(config=True)

    max_copy_folder_size_mb = Int(500, config=True, help="The max folder size that can be copied")

    @default("root_dir")
    def _default_root_dir(self):
        if not self.parent:
            return os.getcwd()
        return self.parent.root_dir

    @validate("root_dir")
    def _validate_root_dir(self, proposal):
        value = proposal["value"]
        if not os.path.isabs(value):
            # If we receive a non-absolute path, make it absolute.
            value = os.path.abspath(value)
        if not os.path.isdir(value):
            raise TraitError("%r is not a directory" % value)
        return value

    @default("preferred_dir")
    def _default_preferred_dir(self):
        if not self.parent:
            return ""
        try:
            value = self.parent.preferred_dir
            if value == self.parent.root_dir:
                value = None
        except AttributeError:
            pass
        else:
            if value is not None:
                warnings.warn(
                    "ServerApp.preferred_dir config is deprecated in jupyter-server 2.0. Use FileContentsManager.preferred_dir instead",
                    FutureWarning,
                    stacklevel=3,
                )
                try:
                    path = Path(value)
                    return path.relative_to(self.root_dir).as_posix()
                except ValueError:
                    raise TraitError("%s is outside root contents directory" % value) from None
        return ""

    @validate("preferred_dir")
    def _validate_preferred_dir(self, proposal):
        # It should be safe to pass an API path through this method:
        proposal["value"] = to_api_path(proposal["value"], self.root_dir)
        return super()._validate_preferred_dir(proposal)

    @default("checkpoints_class")
    def _checkpoints_class_default(self):
        return FileCheckpoints

    delete_to_trash = Bool(
        True,
        config=True,
        help="""If True (default), deleting files will send them to the
        platform's trash/recycle bin, where they can be recovered. If False,
        deleting files really deletes them.""",
    )

    always_delete_dir = Bool(
        False,
        config=True,
        help="""If True, deleting a non-empty directory will always be allowed.
        WARNING this may result in files being permanently removed; e.g. on Windows,
        if the data size is too big for the trash/recycle bin the directory will be permanently
        deleted. If False (default), the non-empty directory will be sent to the trash only
        if safe. And if ``delete_to_trash`` is True, the directory won't be deleted.""",
    )

    listing_page_size_max = Int(
        10000,
        config=True,
        help="""Upper bound on the page_size accepted for paginated directory listings.""",
    )

    listing_cursor_ttl = Int(
        600,
        config=True,
        help="""Seconds a paginated-listing snapshot cursor stays valid after its
        last use. Expired cursors are rejected with an explicit 410 error.""",
    )

    listing_max_cursors = Int(
        128,
        config=True,
        help="""Maximum number of live paginated-listing snapshots kept in memory;
        the least recently used snapshots are evicted beyond this bound.""",
    )

    listing_max_snapshot_entries = Int(
        1000000,
        config=True,
        help="""Maximum number of entries a single paginated-listing snapshot may
        hold. Larger directories are rejected with an explicit 413 error so that
        snapshot memory stays bounded.""",
    )

    @property
    def _listing_store(self):
        """分页快照仓库(按管理器实例惰性创建,仅驻留内存)。

        服务重启后仓库随进程消失,旧游标会得到明确的 server_restarted 错误。
        """
        store = self.__dict__.get("_listing_cursor_store")
        if store is None:
            store = ListingCursorStore(
                ttl=self.listing_cursor_ttl,
                max_cursors=self.listing_max_cursors,
                max_entries=self.listing_max_snapshot_entries,
            )
            self.__dict__["_listing_cursor_store"] = store
        return store

    @default("files_handler_class")
    def _files_handler_class_default(self):
        return AuthenticatedFileHandler

    @default("files_handler_params")
    def _files_handler_params_default(self):
        return {"path": self.root_dir}

    def is_hidden(self, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        os_path = self._get_os_path(path=path)
        return is_hidden(os_path, self.root_dir)

    def is_writable(self, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        os_path = self._get_os_path(path=path)
        try:
            return os.access(os_path, os.W_OK)
        except OSError:
            self.log.error("Failed to check write permissions on %s", os_path)
            return False

    def file_exists(self, path: str) -> bool | t.Awaitable[bool]:
        """项目内部接口说明。"""
        path = path.strip("/")
        os_path = self._get_os_path(path)
        return os.path.isfile(os_path)

    def dir_exists(self, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        os_path = self._get_os_path(path=path)
        return os.path.isdir(os_path)

    def exists(self, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        os_path = self._get_os_path(path=path)
        return exists(os_path)

    def resolve_path(self, path: str) -> str | None:
        """项目内部接口说明。"""
        # transform OS path to API path
        original_path = Path(path)
        try:
            expanded_path = original_path.expanduser()
        except RuntimeError:
            expanded_path = original_path
        if expanded_path != original_path and expanded_path.is_absolute():
            root = Path(os.path.abspath(self.root_dir))
            expanded_abs = Path(os.path.abspath(expanded_path))
            try:
                expanded_abs.relative_to(root)
            except ValueError:
                return None
            path = str(expanded_abs)
        relative_path = to_api_path(path, self.root_dir)
        # check if the API path is within contents directory
        try:
            os_path = self._get_os_path(path=relative_path)
        except web.HTTPError:
            return None
        if exists(os_path):
            return relative_path
        return None

    def _base_model(self, path):
        """项目内部接口说明。"""
        os_path = self._get_os_path(path)
        info = os.lstat(os_path)

        four_o_four = "file or directory does not exist: %r" % path

        if not self.allow_hidden and is_hidden(os_path, self.root_dir):
            self.log.info("Refusing to serve hidden file or directory %r, via 404 Error", os_path)
            raise web.HTTPError(404, four_o_four)

        try:
            # size of file
            size = info.st_size
        except (ValueError, OSError):
            self.log.warning("Unable to get size.")
            size = None

        try:
            last_modified = tz.utcfromtimestamp(info.st_mtime)
        except (ValueError, OverflowError, OSError):
            # Files can rarely have an invalid timestamp
            # https://github.com/jupyter/notebook/issues/2539
            # https://github.com/jupyter/notebook/issues/2757
            # Use the Unix epoch as a fallback so we don't crash.
            self.log.warning("Invalid mtime %s for %s", info.st_mtime, os_path)
            last_modified = datetime(1970, 1, 1, 0, 0, tzinfo=tz.UTC)

        raw_created = _get_created_timestamp(info)
        try:
            created = tz.utcfromtimestamp(raw_created)
        except (ValueError, OverflowError, OSError):  # See above
            self.log.warning("Invalid creation time %s for %s", raw_created, os_path)
            created = datetime(1970, 1, 1, 0, 0, tzinfo=tz.UTC)

        # Create the base model.
        model = {}
        model["name"] = path.rsplit("/", 1)[-1]
        model["path"] = path
        model["last_modified"] = last_modified
        model["created"] = created
        model["content"] = None
        model["format"] = None
        model["mimetype"] = None
        model["size"] = size
        model["writable"] = self.is_writable(path)
        model["hash"] = None
        model["hash_algorithm"] = None

        return model

    def _dir_model(self, path, content=True):
        """项目内部接口说明。"""
        os_path = self._get_os_path(path)

        four_o_four = "directory does not exist: %r" % path

        if not os.path.isdir(os_path):
            raise web.HTTPError(404, four_o_four)
        elif not self.allow_hidden and is_hidden(os_path, self.root_dir):
            self.log.info("Refusing to serve hidden directory %r, via 404 Error", os_path)
            raise web.HTTPError(404, four_o_four)

        model = self._base_model(path)
        model["type"] = "directory"
        model["size"] = None
        if content:
            model["content"] = contents = []
            os_dir = os_path
            for name in os.listdir(os_dir):
                try:
                    os_path = os.path.join(os_dir, name)
                except UnicodeDecodeError as e:
                    self.log.warning("failed to decode filename '%s': %r", name, e)
                    continue

                try:
                    st = os.lstat(os_path)
                except OSError as e:
                    # skip over broken symlinks in listing
                    if e.errno == errno.ENOENT:
                        self.log.warning("%s doesn't exist", os_path)
                    elif e.errno != errno.EACCES:  # Don't provide clues about protected files
                        self.log.warning("Error stat-ing %s: %r", os_path, e)
                    continue

                if (
                    not stat.S_ISLNK(st.st_mode)
                    and not stat.S_ISREG(st.st_mode)
                    and not stat.S_ISDIR(st.st_mode)
                ):
                    self.log.debug("%s not a regular file", os_path)
                    continue

                try:
                    if self.should_list(name) and (
                        self.allow_hidden or not is_file_hidden(os_path, stat_res=st)
                    ):
                        contents.append(self.get(path=f"{path}/{name}", content=False))
                except OSError as e:
                    # ELOOP: recursive symlink, also don't show failure due to permissions
                    if e.errno not in [errno.ELOOP, errno.EACCES]:
                        self.log.warning(
                            "Unknown error checking if file %r is hidden",
                            os_path,
                            exc_info=True,
                        )

            model["format"] = "json"

        return model

    def _file_model(self, path, content=True, format=None, require_hash=False):
        """项目内部接口说明。"""
        model = self._base_model(path)
        model["type"] = "file"

        os_path = self._get_os_path(path)
        model["mimetype"] = mimetypes.guess_type(os_path)[0]

        bytes_content = None
        if content:
            content, format, bytes_content = self._read_file(os_path, format, raw=True)  # type: ignore[misc]
            if model["mimetype"] is None:
                default_mime = {
                    "text": "text/plain",
                    "base64": "application/octet-stream",
                }[format]
                model["mimetype"] = default_mime

            model.update(
                content=content,
                format=format,
            )

        if require_hash:
            if bytes_content is None:
                bytes_content, _ = self._read_file(os_path, "byte")  # type: ignore[assignment,misc]
            model.update(**self._get_hash(bytes_content))  # type: ignore[arg-type]

        return model

    def _notebook_model(self, path, content=True, require_hash=False):
        """项目内部接口说明。"""
        model = self._base_model(path)
        model["type"] = "notebook"
        os_path = self._get_os_path(path)

        bytes_content = None
        if content:
            validation_error: dict[str, t.Any] = {}
            nb, bytes_content = self._read_notebook(
                os_path, as_version=4, capture_validation_error=validation_error, raw=True
            )
            self.mark_trusted_cells(nb, path)
            model["content"] = nb
            model["format"] = "json"
            self.validate_notebook_model(model, validation_error)

        if require_hash:
            if bytes_content is None:
                bytes_content, _ = self._read_file(os_path, "byte")  # type: ignore[misc]
            model.update(**self._get_hash(bytes_content))  # type: ignore[arg-type]

        return model

    def get(self, path, content=True, type=None, format=None, require_hash=False):
        """项目内部接口说明。"""
        path = path.strip("/")
        os_path = self._get_os_path(path)
        four_o_four = "file or directory does not exist: %r" % path

        if not self.exists(path):
            raise web.HTTPError(404, four_o_four)

        if not self.allow_hidden and is_hidden(os_path, self.root_dir):
            self.log.info("Refusing to serve hidden file or directory %r, via 404 Error", os_path)
            raise web.HTTPError(404, four_o_four)

        if os.path.isdir(os_path):
            if type not in (None, "directory"):
                raise web.HTTPError(
                    400,
                    f"{path} is a directory, not a {type}",
                    reason="bad type",
                )
            model = self._dir_model(path, content=content)
        elif type == "notebook" or (type is None and path.endswith(".ipynb")):
            model = self._notebook_model(path, content=content, require_hash=require_hash)
        else:
            if type == "directory":
                raise web.HTTPError(400, "%s is not a directory" % path, reason="bad type")
            model = self._file_model(
                path, content=content, format=format, require_hash=require_hash
            )
        self.emit(data={"action": "get", "path": path})
        return model

    def _scan_listing_dir(self, os_dir, sort=None):
        """扫描目录,返回 (全部原始名称, 当前可服务条目名称)。

        可服务 = 常规文件/目录/符号链接、未被 hide_globs 排除、未隐藏。
        权限与过滤策略在每次扫描时实时评估,因此翻页中途权限收紧会
        立即反映为条目不可见。``sort`` 为 "name"/"size"/"last_modified"
        时对可见名称排序(以名称作确定性决胜键);为 None 时不排序。
        """
        raw_names = os.listdir(os_dir)
        visible: list[tuple[t.Any, str]] = []
        for name in raw_names:
            try:
                child_os_path = os.path.join(os_dir, name)
            except UnicodeDecodeError as e:
                self.log.warning("failed to decode filename '%s': %r", name, e)
                continue

            try:
                st = os.lstat(child_os_path)
            except OSError as e:
                # skip over broken symlinks in listing
                if e.errno == errno.ENOENT:
                    self.log.warning("%s doesn't exist", child_os_path)
                elif e.errno != errno.EACCES:  # Don't provide clues about protected files
                    self.log.warning("Error stat-ing %s: %r", child_os_path, e)
                continue

            if (
                not stat.S_ISLNK(st.st_mode)
                and not stat.S_ISREG(st.st_mode)
                and not stat.S_ISDIR(st.st_mode)
            ):
                self.log.debug("%s not a regular file", child_os_path)
                continue

            if not self.should_list(name):
                continue
            if not self.allow_hidden and is_file_hidden(child_os_path, stat_res=st):
                continue

            if sort == "size":
                visible.append(((st.st_size, name), name))
            elif sort == "last_modified":
                visible.append(((st.st_mtime, name), name))
            else:
                visible.append((name, name))

        if sort is not None:
            visible.sort(key=lambda item: item[0])
        return raw_names, [name for _, name in visible]

    def _validate_listing_request(self, path, sort, page_size):
        """校验分页请求的公共前提,返回目录的 OS 路径。"""
        if sort is not None and sort not in ("name", "size", "last_modified"):
            raise web.HTTPError(
                400,
                f"Invalid listing sort {sort!r}; expected one of 'name', 'size', 'last_modified'.",
            )
        if page_size is not None and not 1 <= page_size <= self.listing_page_size_max:
            raise web.HTTPError(
                400,
                f"page_size must be between 1 and {self.listing_page_size_max}, got {page_size}.",
            )
        os_path = self._get_os_path(path)
        four_o_four = "directory does not exist: %r" % path
        if not os.path.isdir(os_path):
            raise web.HTTPError(404, four_o_four)
        if not self.allow_hidden and is_hidden(os_path, self.root_dir):
            self.log.info("Refusing to serve hidden directory %r, via 404 Error", os_path)
            raise web.HTTPError(404, four_o_four)
        return os_path

    def get_listing_page(
        self,
        path,
        page_size=None,
        cursor=None,
        sort=None,
        require_hash=False,
        hash_budget=None,
    ):
        """按有界快照游标返回目录清单的一页。

        首页(无 ``cursor``)固定排序依据与可见水位(快照);后续页用首页
        签发的游标定位,重新扫描目录以识别删除/新增/权限收紧等破坏连续性
        的变化,并按当前权限即时隐藏条目。哈希等昂贵元数据按
        ``hash_budget`` 预算计算,超出预算的条目留空待客户端按需补取。
        """
        path = path.strip("/")
        os_path = self._validate_listing_request(path, sort, page_size)

        store = self._listing_store
        try:
            if cursor is None:
                if page_size is None:
                    raise web.HTTPError(400, "page_size is required to start a paginated listing.")
                sort = sort or "name"
                raw_names, visible_names = self._scan_listing_dir(os_path, sort=sort)
                snapshot = store.create(
                    path=path, names=visible_names, sort=sort, page_size=page_size
                )
                page_index = 0
            else:
                snapshot, page_index = store.resolve(
                    cursor, path=path, sort=sort, page_size=page_size
                )
                raw_names, visible_names = self._scan_listing_dir(os_path)
        except PermissionError as e:
            raise web.HTTPError(403, f"Permission denied while listing directory: {path!r}") from e

        start = page_index * snapshot.page_size
        page_names = snapshot.names[start : start + snapshot.page_size]
        visible_set = set(visible_names)

        content = []
        hashes_computed = 0
        for name in page_names:
            if name not in visible_set:
                # 快照后被删除或权限收紧不再可见;在 changes 中统一通报。
                continue
            child_path = f"{path}/{name}" if path else name
            want_hash = require_hash and (hash_budget is None or hashes_computed < hash_budget)
            try:
                entry = self.get(path=child_path, content=False, require_hash=want_hash)
            except web.HTTPError as e:
                if e.status_code == 404:
                    # 与并发删除/隐藏竞争失败,跳过;在 changes 中通报。
                    continue
                raise
            except OSError:
                continue
            if want_hash and entry.get("hash") is not None:
                hashes_computed += 1
            content.append(entry)

        hashes_deferred = (
            sum(1 for e in content if e["type"] != "directory" and e["hash"] is None)
            if require_hash
            else 0
        )

        model = self._dir_model(path, content=False)
        model["content"] = content
        model["format"] = "json"
        model["pagination"] = build_pagination_payload(
            store,
            snapshot,
            page_index,
            raw_names=raw_names,
            visible_names=visible_names,
            require_hash=require_hash,
            hash_budget=hash_budget,
            hashes_computed=hashes_computed,
            hashes_deferred=hashes_deferred,
        )
        self.emit(data={"action": "get", "path": path})
        return model

    def _save_directory(self, os_path, model, path=""):
        """项目内部接口说明。"""
        if not self.allow_hidden and is_hidden(os_path, self.root_dir):
            raise web.HTTPError(400, "Cannot create directory %r" % os_path)
        if not os.path.exists(os_path):
            with self.perm_to_403():
                os.mkdir(os_path)
        elif not os.path.isdir(os_path):
            raise web.HTTPError(400, "Not a directory: %s" % (os_path))
        else:
            self.log.debug("Directory %r already exists", os_path)

    def save(self, model, path=""):
        """项目内部接口说明。"""
        path = path.strip("/")

        self.run_pre_save_hooks(model=model, path=path)

        if "type" not in model:
            raise web.HTTPError(400, "No file type provided")
        if "content" not in model and model["type"] != "directory":
            raise web.HTTPError(400, "No file content provided")
        os_path = self._get_os_path(path)

        if not self.allow_hidden and is_hidden(os_path, self.root_dir):
            raise web.HTTPError(400, f"Cannot create file or directory {os_path!r}")

        self.log.debug("Saving %s", os_path)

        validation_error: dict[str, t.Any] = {}
        try:
            if model["type"] == "notebook":
                nb = nbformat.from_dict(model["content"])
                self.check_and_sign(nb, path)
                self._save_notebook(os_path, nb, capture_validation_error=validation_error)
                # One checkpoint should always exist for notebooks.
                if not self.checkpoints.list_checkpoints(path):
                    self.create_checkpoint(path)
            elif model["type"] == "file":
                # Missing format will be handled internally by _save_file.
                self._save_file(os_path, model["content"], model.get("format"))
            elif model["type"] == "directory":
                self._save_directory(os_path, model, path)
            else:
                raise web.HTTPError(400, "Unhandled contents type: %s" % model["type"])
        except web.HTTPError:
            raise
        except Exception as e:
            self.log.error("Error while saving file: %s %s", path, e, exc_info=True)
            raise web.HTTPError(500, f"Unexpected error while saving file: {path} {e}") from e

        validation_message = None
        if model["type"] == "notebook":
            self.validate_notebook_model(model, validation_error=validation_error)
            validation_message = model.get("message", None)

        model = self.get(path, content=False)
        if validation_message:
            model["message"] = validation_message

        self.run_post_save_hooks(model=model, os_path=os_path)
        self.emit(data={"action": "save", "path": path})
        return model

    def delete_file(self, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        os_path = self._get_os_path(path)
        rm = os.unlink

        if not self.allow_hidden and is_hidden(os_path, self.root_dir):
            raise web.HTTPError(400, f"Cannot delete file or directory {os_path!r}")

        four_o_four = "file or directory does not exist: %r" % path
        if not self.exists(path):
            raise web.HTTPError(404, four_o_four)

        def is_non_empty_dir(os_path):
            if os.path.isdir(os_path):
                # A directory containing only leftover checkpoints is
                # considered empty.
                cp_dir = getattr(self.checkpoints, "checkpoint_dir", None)
                if set(os.listdir(os_path)) - {cp_dir}:
                    return True

            return False

        if self.delete_to_trash:
            if not self.always_delete_dir and sys.platform == "win32" and is_non_empty_dir(os_path):
                # send2trash can really delete files on Windows, so disallow
                # deleting non-empty files. See Github issue 3631.
                raise web.HTTPError(400, "Directory %s not empty" % os_path)
            # send2trash now supports deleting directories. see #1290
            if not self.is_writable(path):
                raise web.HTTPError(403, "Permission denied: %s" % path) from None
            self.log.debug("Sending %s to trash", os_path)
            try:
                send2trash(os_path)
            except OSError as e:
                raise web.HTTPError(400, "send2trash failed: %s" % e) from e
            return

        if os.path.isdir(os_path):
            # Don't permanently delete non-empty directories.
            if not self.always_delete_dir and is_non_empty_dir(os_path):
                raise web.HTTPError(400, "Directory %s not empty" % os_path)
            self.log.debug("Removing directory %s", os_path)
            with self.perm_to_403():
                shutil.rmtree(os_path)
        else:
            self.log.debug("Unlinking file %s", os_path)
            with self.perm_to_403():
                rm(os_path)

    def rename_file(self, old_path, new_path):
        """项目内部接口说明。"""
        old_path = old_path.strip("/")
        new_path = new_path.strip("/")
        if new_path == old_path:
            return

        new_os_path = self._get_os_path(new_path)
        old_os_path = self._get_os_path(old_path)

        if not self.allow_hidden and (
            is_hidden(old_os_path, self.root_dir) or is_hidden(new_os_path, self.root_dir)
        ):
            raise web.HTTPError(400, f"Cannot rename file or directory {old_os_path!r}")

        # Should we proceed with the move?
        if os.path.exists(new_os_path) and not samefile(old_os_path, new_os_path):
            raise web.HTTPError(409, "File already exists: %s" % new_path)

        # Move the file
        try:
            with self.perm_to_403():
                shutil.move(old_os_path, new_os_path)
        except web.HTTPError:
            raise
        except FileNotFoundError:
            raise web.HTTPError(404, f"File or directory does not exist: {old_path}") from None
        except Exception as e:
            raise web.HTTPError(500, f"Unknown error renaming file: {old_path} {e}") from e

    def info_string(self):
        """项目内部接口说明。"""
        return _i18n("Serving notebooks from local directory: %s") % self.root_dir

    def get_kernel_path(self, path, model=None):
        """项目内部接口说明。"""
        if self.dir_exists(path):
            return path
        parent_dir = path.rsplit("/", 1)[0] if "/" in path else ""
        return parent_dir

    def copy(self, from_path, to_path=None):
        """项目内部接口说明。"""
        to_path_original = str(to_path)
        path = from_path.strip("/")
        if to_path is not None:
            to_path = to_path.strip("/")

        if "/" in path:
            from_dir, from_name = path.rsplit("/", 1)
        else:
            from_dir = ""
            from_name = path

        model = self.get(path)
        # limit the size of folders being copied to prevent a timeout error
        if model["type"] == "directory":
            self.check_folder_size(path)
        else:
            # let the super class handle copying files
            return super().copy(from_path=from_path, to_path=to_path)

        is_destination_specified = to_path is not None
        to_name = copy_pat.sub(".", from_name)
        if not is_destination_specified:
            to_path = from_dir
        if self.dir_exists(to_path):
            name = copy_pat.sub(".", from_name)
            to_name = super().increment_filename(name, to_path, insert="-Copy")
        to_path = f"{to_path}/{to_name}"

        return self._copy_dir(
            from_path=from_path,
            to_path_original=to_path_original,
            to_name=to_name,
            to_path=to_path,
        )

    def _copy_dir(self, from_path, to_path_original, to_name, to_path):
        """项目内部接口说明。"""
        try:
            os_from_path = self._get_os_path(from_path.strip("/"))
            os_to_path = f"{self._get_os_path(to_path_original.strip('/'))}/{to_name}"
            shutil.copytree(os_from_path, os_to_path)
            model = self.get(to_path, content=False)
        except OSError as err:
            self.log.error(f"OSError in _copy_dir: {err}")
            raise web.HTTPError(
                400,
                f"Can't copy '{from_path}' into Folder '{to_path}'",
            ) from err

        return model

    def check_folder_size(self, path):
        """项目内部接口说明。"""
        limit_bytes = self.max_copy_folder_size_mb * 1024 * 1024
        size = int(self._get_dir_size(self._get_os_path(path)))
        # convert from KB to Bytes for macOS
        size = size * 1024 if platform.system() == "Darwin" else size

        if size > limit_bytes:
            raise web.HTTPError(
                400,
                f"""
                    Can't copy folders larger than {self.max_copy_folder_size_mb}MB,
                    "{path}" is {self._human_readable_size(size)}
                """,
            )

    def _get_dir_size(self, path="."):
        """项目内部接口说明。"""
        try:
            if platform.system() == "Darwin":
                # returns the size of the folder in KB
                result = subprocess.run(
                    ["du", "-sk", path],  # noqa: S607
                    capture_output=True,
                    check=True,
                ).stdout.split()
            else:
                result = subprocess.run(
                    ["du", "-s", "--block-size=1", path],  # noqa: S607
                    capture_output=True,
                    check=True,
                ).stdout.split()

            self.log.info(f"current status of du command {result}")
            size = result[0].decode("utf-8")
        except Exception:
            self.log.warning(
                "Not able to get the size of the %s directory. Copying might be slow if the directory is large!",
                path,
            )
            return "0"
        return size

    def _human_readable_size(self, size):
        """项目内部接口说明。"""
        if size == 0:
            return "0 Bytes"

        units = ["Bytes", "KB", "MB", "GB", "TB", "PB"]
        order = int(math.log2(size) / 10) if size else 0

        return f"{size / (1 << (order * 10)):.4g} {units[order]}"


class AsyncFileContentsManager(  # type: ignore[misc]
    FileContentsManager, AsyncFileManagerMixin, AsyncContentsManager
):
    """项目内部接口说明。"""

    @default("checkpoints_class")
    def _checkpoints_class_default(self):
        return AsyncFileCheckpoints

    async def _dir_model(self, path, content=True):
        """项目内部接口说明。"""
        os_path = self._get_os_path(path)

        four_o_four = "directory does not exist: %r" % path

        if not os.path.isdir(os_path):
            raise web.HTTPError(404, four_o_four)
        elif not self.allow_hidden and is_hidden(os_path, self.root_dir):
            self.log.info("Refusing to serve hidden directory %r, via 404 Error", os_path)
            raise web.HTTPError(404, four_o_four)

        model = self._base_model(path)
        model["type"] = "directory"
        model["size"] = None
        if content:
            model["content"] = contents = []
            os_dir = os_path
            dir_contents = await run_sync(os.listdir, os_dir)
            for name in dir_contents:
                try:
                    os_path = os.path.join(os_dir, name)
                except UnicodeDecodeError as e:
                    self.log.warning("failed to decode filename '%s': %r", name, e)
                    continue

                try:
                    st = await run_sync(os.lstat, os_path)
                except OSError as e:
                    # skip over broken symlinks in listing
                    if e.errno == errno.ENOENT:
                        self.log.warning("%s doesn't exist", os_path)
                    elif e.errno != errno.EACCES:  # Don't provide clues about protected files
                        self.log.warning("Error stat-ing %s: %r", os_path, e)
                    continue

                if (
                    not stat.S_ISLNK(st.st_mode)
                    and not stat.S_ISREG(st.st_mode)
                    and not stat.S_ISDIR(st.st_mode)
                ):
                    self.log.debug("%s not a regular file", os_path)
                    continue

                try:
                    if self.should_list(name) and (
                        self.allow_hidden or not is_file_hidden(os_path, stat_res=st)
                    ):
                        contents.append(await self.get(path=f"{path}/{name}", content=False))
                except OSError as e:
                    # ELOOP: recursive symlink, also don't show failure due to permissions
                    if e.errno not in [errno.ELOOP, errno.EACCES]:
                        self.log.warning(
                            "Unknown error checking if file %r is hidden",
                            os_path,
                            exc_info=True,
                        )

            model["format"] = "json"

        return model

    async def _file_model(self, path, content=True, format=None, require_hash=False):
        """项目内部接口说明。"""
        model = self._base_model(path)
        model["type"] = "file"

        os_path = self._get_os_path(path)
        model["mimetype"] = mimetypes.guess_type(os_path)[0]

        bytes_content = None
        if content:
            content, format, bytes_content = await self._read_file(os_path, format, raw=True)  # type: ignore[misc]
            if model["mimetype"] is None:
                default_mime = {
                    "text": "text/plain",
                    "base64": "application/octet-stream",
                }[format]
                model["mimetype"] = default_mime

            model.update(
                content=content,
                format=format,
            )

        if require_hash:
            if bytes_content is None:
                bytes_content, _ = await self._read_file(os_path, "byte")  # type: ignore[assignment,misc]
            model.update(**self._get_hash(bytes_content))  # type: ignore[arg-type]

        return model

    async def _notebook_model(self, path, content=True, require_hash=False):
        """项目内部接口说明。"""
        model = self._base_model(path)
        model["type"] = "notebook"
        os_path = self._get_os_path(path)

        bytes_content = None
        if content:
            validation_error: dict[str, t.Any] = {}
            nb, bytes_content = await self._read_notebook(
                os_path, as_version=4, capture_validation_error=validation_error, raw=True
            )
            self.mark_trusted_cells(nb, path)
            model["content"] = nb
            model["format"] = "json"
            self.validate_notebook_model(model, validation_error)

        if require_hash:
            if bytes_content is None:
                bytes_content, _ = await self._read_file(os_path, "byte")  # type: ignore[misc]
            model.update(**(self._get_hash(bytes_content)))  # type: ignore[arg-type]

        return model

    async def get(self, path, content=True, type=None, format=None, require_hash=False):
        """项目内部接口说明。"""
        path = path.strip("/")

        if not self.exists(path):
            raise web.HTTPError(404, "No such file or directory: %s" % path)

        os_path = self._get_os_path(path)
        if os.path.isdir(os_path):
            if type not in (None, "directory"):
                raise web.HTTPError(
                    400,
                    f"{path} is a directory, not a {type}",
                    reason="bad type",
                )
            model = await self._dir_model(path, content=content)
        elif type == "notebook" or (type is None and path.endswith(".ipynb")):
            model = await self._notebook_model(path, content=content, require_hash=require_hash)
        else:
            if type == "directory":
                raise web.HTTPError(400, "%s is not a directory" % path, reason="bad type")
            model = await self._file_model(
                path, content=content, format=format, require_hash=require_hash
            )
        self.emit(data={"action": "get", "path": path})
        return model

    async def get_listing_page(
        self,
        path,
        page_size=None,
        cursor=None,
        sort=None,
        require_hash=False,
        hash_budget=None,
    ):
        """按有界快照游标返回目录清单的一页(异步版本)。

        语义与同步版一致:首页固定排序依据与可见水位,后续页识别破坏
        连续性的目录变化,权限收紧立即隐藏条目,哈希按预算延迟计算。
        目录扫描一次性放入线程执行,避免逐条目 await 的额外开销。
        """
        path = path.strip("/")
        os_path = self._validate_listing_request(path, sort, page_size)

        store = self._listing_store
        try:
            if cursor is None:
                if page_size is None:
                    raise web.HTTPError(400, "page_size is required to start a paginated listing.")
                sort = sort or "name"
                raw_names, visible_names = await run_sync(self._scan_listing_dir, os_path, sort)
                snapshot = store.create(
                    path=path, names=visible_names, sort=sort, page_size=page_size
                )
                page_index = 0
            else:
                snapshot, page_index = store.resolve(
                    cursor, path=path, sort=sort, page_size=page_size
                )
                raw_names, visible_names = await run_sync(self._scan_listing_dir, os_path)
        except PermissionError as e:
            raise web.HTTPError(403, f"Permission denied while listing directory: {path!r}") from e

        start = page_index * snapshot.page_size
        page_names = snapshot.names[start : start + snapshot.page_size]
        visible_set = set(visible_names)

        content = []
        hashes_computed = 0
        for name in page_names:
            if name not in visible_set:
                # 快照后被删除或权限收紧不再可见;在 changes 中统一通报。
                continue
            child_path = f"{path}/{name}" if path else name
            want_hash = require_hash and (hash_budget is None or hashes_computed < hash_budget)
            try:
                entry = await self.get(path=child_path, content=False, require_hash=want_hash)
            except web.HTTPError as e:
                if e.status_code == 404:
                    # 与并发删除/隐藏竞争失败,跳过;在 changes 中通报。
                    continue
                raise
            except OSError:
                continue
            if want_hash and entry.get("hash") is not None:
                hashes_computed += 1
            content.append(entry)

        hashes_deferred = (
            sum(1 for e in content if e["type"] != "directory" and e["hash"] is None)
            if require_hash
            else 0
        )

        model = await self._dir_model(path, content=False)
        model["content"] = content
        model["format"] = "json"
        model["pagination"] = build_pagination_payload(
            store,
            snapshot,
            page_index,
            raw_names=raw_names,
            visible_names=visible_names,
            require_hash=require_hash,
            hash_budget=hash_budget,
            hashes_computed=hashes_computed,
            hashes_deferred=hashes_deferred,
        )
        self.emit(data={"action": "get", "path": path})
        return model

    async def _save_directory(self, os_path, model, path=""):
        """项目内部接口说明。"""
        if not self.allow_hidden and is_hidden(os_path, self.root_dir):
            raise web.HTTPError(400, "Cannot create hidden directory %r" % os_path)
        if not os.path.exists(os_path):
            with self.perm_to_403():
                await run_sync(os.mkdir, os_path)
        elif not os.path.isdir(os_path):
            raise web.HTTPError(400, "Not a directory: %s" % (os_path))
        else:
            self.log.debug("Directory %r already exists", os_path)

    async def save(self, model, path=""):
        """项目内部接口说明。"""
        path = path.strip("/")

        self.run_pre_save_hooks(model=model, path=path)

        if "type" not in model:
            raise web.HTTPError(400, "No file type provided")
        if "content" not in model and model["type"] != "directory":
            raise web.HTTPError(400, "No file content provided")

        os_path = self._get_os_path(path)
        self.log.debug("Saving %s", os_path)

        validation_error: dict[str, t.Any] = {}
        try:
            if model["type"] == "notebook":
                nb = nbformat.from_dict(model["content"])
                self.check_and_sign(nb, path)
                await self._save_notebook(os_path, nb, capture_validation_error=validation_error)
                # One checkpoint should always exist for notebooks.
                if not (await self.checkpoints.list_checkpoints(path)):
                    await self.create_checkpoint(path)
            elif model["type"] == "file":
                # Missing format will be handled internally by _save_file.
                await self._save_file(os_path, model["content"], model.get("format"))
            elif model["type"] == "directory":
                await self._save_directory(os_path, model, path)
            else:
                raise web.HTTPError(400, "Unhandled contents type: %s" % model["type"])
        except web.HTTPError:
            raise
        except Exception as e:
            self.log.error("Error while saving file: %s %s", path, e, exc_info=True)
            raise web.HTTPError(500, f"Unexpected error while saving file: {path} {e}") from e

        validation_message = None
        if model["type"] == "notebook":
            self.validate_notebook_model(model, validation_error=validation_error)
            validation_message = model.get("message", None)

        model = await self.get(path, content=False)
        if validation_message:
            model["message"] = validation_message

        self.run_post_save_hooks(model=model, os_path=os_path)
        self.emit(data={"action": "save", "path": path})
        return model

    async def delete_file(self, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        os_path = self._get_os_path(path)
        rm = os.unlink

        if not self.allow_hidden and is_hidden(os_path, self.root_dir):
            raise web.HTTPError(400, f"Cannot delete file or directory {os_path!r}")

        if not os.path.exists(os_path):
            raise web.HTTPError(404, "File or directory does not exist: %s" % os_path)

        async def is_non_empty_dir(os_path):
            if os.path.isdir(os_path):
                # A directory containing only leftover checkpoints is
                # considered empty.
                cp_dir = getattr(self.checkpoints, "checkpoint_dir", None)
                dir_contents = set(await run_sync(os.listdir, os_path))
                if dir_contents - {cp_dir}:
                    return True

            return False

        if self.delete_to_trash:
            if (
                not self.always_delete_dir
                and sys.platform == "win32"
                and await is_non_empty_dir(os_path)
            ):
                # send2trash can really delete files on Windows, so disallow
                # deleting non-empty files. See Github issue 3631.
                raise web.HTTPError(400, "Directory %s not empty" % os_path)
            # send2trash now supports deleting directories. see #1290
            if not self.is_writable(path):
                raise web.HTTPError(403, "Permission denied: %s" % path) from None
            self.log.debug("Sending %s to trash", os_path)
            try:
                send2trash(os_path)
            except OSError as e:
                raise web.HTTPError(400, "send2trash failed: %s" % e) from e
            return

        if os.path.isdir(os_path):
            # Don't permanently delete non-empty directories.
            if not self.always_delete_dir and await is_non_empty_dir(os_path):
                raise web.HTTPError(400, "Directory %s not empty" % os_path)
            self.log.debug("Removing directory %s", os_path)
            with self.perm_to_403():
                await run_sync(shutil.rmtree, os_path)
        else:
            self.log.debug("Unlinking file %s", os_path)
            with self.perm_to_403():
                await run_sync(rm, os_path)

    async def rename_file(self, old_path, new_path):
        """项目内部接口说明。"""
        old_path = old_path.strip("/")
        new_path = new_path.strip("/")
        if new_path == old_path:
            return

        new_os_path = self._get_os_path(new_path)
        old_os_path = self._get_os_path(old_path)

        if not self.allow_hidden and (
            is_hidden(old_os_path, self.root_dir) or is_hidden(new_os_path, self.root_dir)
        ):
            raise web.HTTPError(400, f"Cannot rename file or directory {old_os_path!r}")

        # Should we proceed with the move?
        if os.path.exists(new_os_path) and not samefile(old_os_path, new_os_path):
            raise web.HTTPError(409, "File already exists: %s" % new_path)

        # Move the file
        try:
            with self.perm_to_403():
                await run_sync(shutil.move, old_os_path, new_os_path)
        except web.HTTPError:
            raise
        except FileNotFoundError:
            raise web.HTTPError(404, f"File or directory does not exist: {old_path}") from None
        except Exception as e:
            raise web.HTTPError(500, f"Unknown error renaming file: {old_path} {e}") from e

    async def dir_exists(self, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        os_path = self._get_os_path(path=path)
        return os.path.isdir(os_path)

    async def file_exists(self, path: str) -> bool:
        """项目内部接口说明。"""
        path = path.strip("/")
        os_path = self._get_os_path(path)
        return os.path.isfile(os_path)

    async def is_hidden(self, path):
        """项目内部接口说明。"""
        path = path.strip("/")
        os_path = self._get_os_path(path=path)
        return is_hidden(os_path, self.root_dir)

    async def get_kernel_path(self, path, model=None):
        """项目内部接口说明。"""
        if await self.dir_exists(path):
            return path
        parent_dir = path.rsplit("/", 1)[0] if "/" in path else ""
        return parent_dir

    async def copy(self, from_path, to_path=None):
        """项目内部接口说明。"""
        to_path_original = str(to_path)
        path = from_path.strip("/")
        if to_path is not None:
            to_path = to_path.strip("/")

        if "/" in path:
            from_dir, from_name = path.rsplit("/", 1)
        else:
            from_dir = ""
            from_name = path

        model = await self.get(path)
        # limit the size of folders being copied to prevent a timeout error
        if model["type"] == "directory":
            await self.check_folder_size(path)
        else:
            # let the super class handle copying files
            return await AsyncContentsManager.copy(self, from_path=from_path, to_path=to_path)

        is_destination_specified = to_path is not None
        to_name = copy_pat.sub(".", from_name)
        if not is_destination_specified:
            to_path = from_dir
        if await self.dir_exists(to_path):
            name = copy_pat.sub(".", from_name)
            to_name = await super().increment_filename(name, to_path, insert="-Copy")
        to_path = f"{to_path}/{to_name}"

        return await self._copy_dir(
            from_path=from_path,
            to_path_original=to_path_original,
            to_name=to_name,
            to_path=to_path,
        )

    async def _copy_dir(
        self, from_path: str, to_path_original: str, to_name: str, to_path: str
    ) -> dict[str, t.Any]:
        """项目内部接口说明。"""
        try:
            os_from_path = self._get_os_path(from_path.strip("/"))
            os_to_path = f"{self._get_os_path(to_path_original.strip('/'))}/{to_name}"
            shutil.copytree(os_from_path, os_to_path)
            model = await self.get(to_path, content=False)
        except OSError as err:
            self.log.error(f"OSError in _copy_dir: {err}")
            raise web.HTTPError(
                400,
                f"Can't copy '{from_path}' into read-only Folder '{to_path}'",
            ) from err

        return model  # type:ignore[no-any-return]

    async def check_folder_size(self, path: str) -> None:
        """项目内部接口说明。"""
        limit_bytes = self.max_copy_folder_size_mb * 1024 * 1024

        size = int(await self._get_dir_size(self._get_os_path(path)))
        # convert from KB to Bytes for macOS
        size = size * 1024 if platform.system() == "Darwin" else size
        if size > limit_bytes:
            raise web.HTTPError(
                400,
                f"""
                    Can't copy folders larger than {self.max_copy_folder_size_mb}MB,
                    "{path}" is {await self._human_readable_size(size)}
                """,
            )

    async def _get_dir_size(self, path: str = ".") -> str:
        """项目内部接口说明。"""
        try:
            if platform.system() == "Darwin":
                # returns the size of the folder in KB
                args = ["-sk", path]
            else:
                args = ["-s", "--block-size=1", path]
            proc = await asyncio.create_subprocess_exec(
                "du", *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, _ = await proc.communicate()
            result = await proc.wait()
            self.log.info(f"current status of du command {result}")
            assert result == 0
            size = stdout.decode("utf-8").split()[0]
        except Exception:
            self.log.warning(
                "Not able to get the size of the %s directory. Copying might be slow if the directory is large!",
                path,
            )
            return "0"
        return size

    async def _human_readable_size(self, size: int) -> str:
        """项目内部接口说明。"""
        if size == 0:
            return "0 Bytes"

        units = ["Bytes", "KB", "MB", "GB", "TB", "PB"]
        order = int(math.log2(size) / 10) if size else 0

        return f"{size / (1 << (order * 10)):.4g} {units[order]}"
