"""目录分页读取的有界快照游标。

问题背景:工作区目录达到几十万项后,一次性生成全部条目的元数据既耗费内存,
又会在并发增删时让简单分页(offset/limit)产生重复与遗漏。

本模块提供"有界快照游标":

- 首页请求时对待服务条目做一次过滤与排序,把*排序后的名称列表*(而非完整
  元数据)存入有界的内存快照仓库;名称列表即"可见水位",此后新增的条目
  不会进入后续页面,从源头避免错位。
- 快照仓库有界:TTL 过期 + LRU 驱逐 + 单快照条目上限,服务器内存不随
  客户端行为无界增长。
- 游标令牌内嵌进程启动纪元(BOOT_ID),服务重启后旧游标返回明确的
  ``server_restarted`` 错误,而不是含糊的"未找到"。
- 令牌本身编码(快照 id, 页码),同一页重试使用同一令牌,结果确定。
"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import time
import typing as t
import uuid
from collections import OrderedDict
from dataclasses import dataclass

#: 游标令牌线格式版本。
CURSOR_VERSION = "v1"

#: 进程启动纪元。快照只保存在内存中,服务重启后全部失效;
#: 把启动纪元编入令牌,可以明确地区分"服务已重启"与"游标过期"。
BOOT_ID = uuid.uuid4().hex

#: 每类变化最多回报的名称条数,防止变化清单本身无界。
MAX_REPORTED_CHANGES = 100


class ListingCursorError(Exception):
    """分页游标无法兑现时抛出的异常,携带明确的机器可读原因。

    Attributes
    ----------
    reason : str
        稳定的机器可读错误码,如 ``cursor_expired``、``server_restarted``、
        ``cursor_malformed``、``cursor_mismatch``、``snapshot_too_large``。
    status_code : int
        处理器应返回的 HTTP 状态码。
    """

    def __init__(self, reason: str, message: str, status_code: int = 410):
        super().__init__(message)
        self.reason = reason
        self.status_code = status_code


@dataclass
class ListingSnapshot:
    """单个目录的一次有界快照。

    只保存过滤、排序后的条目*名称*(可见水位),不保存任何昂贵元数据;
    每页的条目模型在取页时按需构建。
    """

    snapshot_id: str
    path: str
    names: tuple[str, ...]  # 快照时刻可服务条目的名称,按 sort 排序
    sort: str
    page_size: int
    created: float
    expires: float

    @property
    def total(self) -> int:
        """快照中的条目总数(可见水位)。"""
        return len(self.names)

    @property
    def page_count(self) -> int:
        """总页数,空目录按 1 页(空页)计。"""
        return max(1, -(-self.total // self.page_size))


class ListingCursorStore:
    """有界的快照仓库:TTL 过期 + LRU 驱逐 + 单快照条目上限。

    参数中的 ``clock`` 可注入,便于测试;默认使用单调时钟,
    不受系统时间回拨影响。
    """

    def __init__(
        self,
        ttl: float = 600.0,
        max_cursors: int = 128,
        max_entries: int = 1_000_000,
        clock: t.Callable[[], float] = time.monotonic,
    ):
        self.ttl = float(ttl)
        self.max_cursors = int(max_cursors)
        self.max_entries = int(max_entries)
        self._clock = clock
        self._snapshots: OrderedDict[str, ListingSnapshot] = OrderedDict()

    def __len__(self) -> int:
        return len(self._snapshots)

    def _purge_expired(self) -> None:
        now = self._clock()
        expired = [sid for sid, snap in self._snapshots.items() if snap.expires <= now]
        for sid in expired:
            del self._snapshots[sid]

    def time_remaining(self, snapshot: ListingSnapshot) -> float:
        """快照距过期的剩余秒数(不小于 0)。"""
        return max(0.0, snapshot.expires - self._clock())

    def create(
        self,
        *,
        path: str,
        names: t.Sequence[str],
        sort: str,
        page_size: int,
    ) -> ListingSnapshot:
        """以过滤排序后的名称列表创建快照,并执行有界约束。"""
        if len(names) > self.max_entries:
            raise ListingCursorError(
                "snapshot_too_large",
                f"Directory {path!r} has {len(names)} visible entries, which exceeds "
                f"the snapshot limit of {self.max_entries}; narrow the directory or "
                "raise the listing_max_snapshot_entries limit.",
                status_code=413,
            )
        self._purge_expired()
        while len(self._snapshots) >= self.max_cursors:
            # 驱逐最久未使用的快照,为新快照腾出位置。
            self._snapshots.popitem(last=False)
        now = self._clock()
        snapshot = ListingSnapshot(
            snapshot_id=uuid.uuid4().hex,
            path=path,
            names=tuple(names),
            sort=sort,
            page_size=page_size,
            created=now,
            expires=now + self.ttl,
        )
        self._snapshots[snapshot.snapshot_id] = snapshot
        return snapshot

    def make_token(self, snapshot: ListingSnapshot, page_index: int) -> str:
        """生成指向快照某一页的游标令牌(同一令牌可安全重试)。"""
        return f"{CURSOR_VERSION}.{BOOT_ID}.{snapshot.snapshot_id}.{page_index}"

    def resolve(
        self,
        token: str,
        *,
        path: str,
        sort: str | None = None,
        page_size: int | None = None,
    ) -> tuple[ListingSnapshot, int]:
        """把游标令牌解析为 (快照, 页码)。

        成功解析会刷新快照的过期时间(滑动过期)并更新 LRU 顺序;
        任何不一致都抛出带明确原因的 :class:`ListingCursorError`。
        """
        parts = str(token).split(".")
        if len(parts) != 4 or parts[0] != CURSOR_VERSION:
            raise ListingCursorError(
                "cursor_malformed",
                f"Malformed listing cursor {token!r}; start the listing again to get a fresh one.",
                status_code=400,
            )
        _, boot_id, snapshot_id, page_index_str = parts
        if boot_id != BOOT_ID:
            raise ListingCursorError(
                "server_restarted",
                "The listing cursor was issued before the server restarted; "
                "start the listing again.",
            )
        try:
            page_index = int(page_index_str)
        except ValueError:
            raise ListingCursorError(
                "cursor_malformed",
                f"Malformed listing cursor {token!r}; start the listing again to get a fresh one.",
                status_code=400,
            ) from None
        self._purge_expired()
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is None:
            raise ListingCursorError(
                "cursor_expired",
                "The listing cursor has expired or was evicted; start the listing again.",
            )
        if (
            path != snapshot.path
            or (sort is not None and sort != snapshot.sort)
            or (page_size is not None and page_size != snapshot.page_size)
        ):
            raise ListingCursorError(
                "cursor_mismatch",
                "The listing cursor does not match the requested path, sort order or "
                "page size; these are fixed by the first page and cannot change "
                "mid-listing.",
                status_code=400,
            )
        if not 0 <= page_index < snapshot.page_count:
            raise ListingCursorError(
                "cursor_mismatch",
                f"Page index {page_index} is out of range for a listing of "
                f"{snapshot.page_count} page(s).",
                status_code=400,
            )
        # 滑动过期 + LRU 触碰:活跃翻页不会因 TTL 中断,闲置游标按期回收。
        snapshot.expires = self._clock() + self.ttl
        self._snapshots.move_to_end(snapshot_id)
        return snapshot, page_index


def compute_changes(
    snapshot_names: t.Iterable[str],
    current_visible: t.Iterable[str],
    current_raw: t.Iterable[str],
    max_reported: int = MAX_REPORTED_CHANGES,
) -> dict[str, t.Any]:
    """对比快照与当前目录扫描,识别会破坏翻页连续性的变化。

    - ``deleted``:快照中存在、磁盘上已消失(删除/改名)的条目——已翻过的
      页面里可能含有它们,连续性被破坏,``continuity`` 置为 ``"dirty"``。
    - ``hidden``:仍在磁盘上、但按当前权限/过滤策略不再可见的条目——
      权限收紧立即生效,同样置为 ``"dirty"``。
    - ``added``:快照之后才出现的条目——位于可见水位之外,不会被后续页面
      返回,仅作通报,不影响连续性。
    """
    snap = set(snapshot_names)
    visible = set(current_visible)
    raw = set(current_raw)
    deleted = sorted(snap - raw)
    hidden = sorted((snap & raw) - visible)
    added = sorted(visible - snap)
    truncated = any(len(changes) > max_reported for changes in (deleted, hidden, added))
    return {
        "continuity": "dirty" if (deleted or hidden) else "clean",
        "deleted_count": len(deleted),
        "deleted": deleted[:max_reported],
        "hidden_count": len(hidden),
        "hidden": hidden[:max_reported],
        "added_count": len(added),
        "added": added[:max_reported],
        "truncated": truncated,
    }


def build_pagination_payload(
    store: ListingCursorStore,
    snapshot: ListingSnapshot,
    page_index: int,
    *,
    raw_names: t.Iterable[str],
    visible_names: t.Iterable[str],
    require_hash: bool,
    hash_budget: int | None,
    hashes_computed: int,
    hashes_deferred: int,
) -> dict[str, t.Any]:
    """组装目录模型上的 ``pagination`` 负载。"""
    # 空目录只有一页;非空目录在最后一页之后没有更多内容。
    has_more = snapshot.total > 0 and page_index + 1 < snapshot.page_count
    return {
        "cursor": store.make_token(snapshot, page_index + 1) if has_more else None,
        "page_index": page_index,
        "page_size": snapshot.page_size,
        "page_count": snapshot.page_count,
        "total_entries": snapshot.total,
        "has_more": has_more,
        "sort": snapshot.sort,
        "expires_in": round(store.time_remaining(snapshot), 3),
        "changes": compute_changes(snapshot.names, visible_names, raw_names),
        "hash": {
            "requested": bool(require_hash),
            "budget": hash_budget,
            "computed": hashes_computed,
            "deferred": hashes_deferred,
        },
    }
