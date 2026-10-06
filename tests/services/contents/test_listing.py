"""目录分页读取(有界快照游标)的测试。

覆盖:整目录遍历无重复/遗漏、并发增删与权限收紧的变化识别、
哈希预算延迟计算、游标过期/服务重启/同页重试/篡改的明确结果,
以及不分页调用的兼容性。
"""

import json
import os
import warnings

import pytest
import tornado

from jupyter_server.services.contents import listing
from jupyter_server.services.contents.listing import (
    ListingCursorError,
    ListingCursorStore,
    compute_changes,
)


@pytest.fixture(autouse=True)
def suppress_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The synchronous ContentsManager",
            category=DeprecationWarning,
        )
        yield


@pytest.fixture(params=["FileContentsManager", "AsyncFileContentsManager"])
def jp_argv(request):
    return [
        "--ServerApp.contents_manager_class=jupyter_server.services.contents.filemanager."
        + request.param
    ]


@pytest.fixture
def listing_dir(jp_root_dir):
    """创建一个条目确定的目录:25 个文件 + 1 个子目录。"""
    dir_path = jp_root_dir / "paginate_me"
    dir_path.mkdir()
    for i in range(25):
        (dir_path / f"file{i:02d}.txt").write_text(f"content {i}", encoding="utf-8")
    (dir_path / "subdir").mkdir()
    return "paginate_me"


def expected_names():
    return sorted([f"file{i:02d}.txt" for i in range(25)] + ["subdir"])


async def fetch_page(jp_fetch, path, **params):
    r = await jp_fetch(
        "api", "contents", path, method="GET", params={k: str(v) for k, v in params.items()}
    )
    return json.loads(r.body.decode())


async def fetch_error(jp_fetch, path, **params):
    """发起请求并返回 (HTTP 状态码, 错误响应体);非 JSON 响应体时返回空字典。"""
    with pytest.raises(tornado.httpclient.HTTPClientError) as exc_info:
        await fetch_page(jp_fetch, path, **params)
    err = exc_info.value
    body = {}
    if err.response is not None and err.response.body:
        try:
            body = json.loads(err.response.body.decode())
        except ValueError:
            pass
    return err.code, body


async def walk_listing(jp_fetch, path, **first_params):
    """跟随 cursor 翻页直到结束,返回 (所有页模型, 全部条目名)。"""
    pages = []
    params = dict(first_params)
    for _ in range(1000):
        model = await fetch_page(jp_fetch, path, **params)
        pages.append(model)
        cursor = model["pagination"]["cursor"]
        if cursor is None:
            assert model["pagination"]["has_more"] is False
            return pages, [e["name"] for page in pages for e in page["content"]]
        assert model["pagination"]["has_more"] is True
        params = {"cursor": cursor}
    raise AssertionError("pagination did not terminate")


# ---------------------------------------------------------------------------
# 基本分页行为
# ---------------------------------------------------------------------------


async def test_walk_all_pages_no_duplicates_or_omissions(jp_fetch, listing_dir):
    pages, names = await walk_listing(jp_fetch, listing_dir, page_size=10)
    assert len(pages) == 3
    assert [len(p["content"]) for p in pages] == [10, 10, 6]
    # 与整目录读取完全一致:无重复、无遗漏
    assert names == expected_names()
    assert len(set(names)) == 26
    for index, page in enumerate(pages):
        pagination = page["pagination"]
        assert pagination["page_index"] == index
        assert pagination["page_size"] == 10
        assert pagination["page_count"] == 3
        assert pagination["total_entries"] == 26
        assert pagination["sort"] == "name"
        assert pagination["changes"]["continuity"] == "clean"
        assert pagination["expires_in"] > 0
    assert pages[-1]["pagination"]["cursor"] is None


async def test_empty_directory_is_a_single_empty_page(jp_fetch, jp_root_dir):
    (jp_root_dir / "empty_dir").mkdir()
    model = await fetch_page(jp_fetch, "empty_dir", page_size=5)
    assert model["content"] == []
    pagination = model["pagination"]
    assert pagination["total_entries"] == 0
    assert pagination["page_count"] == 1
    assert pagination["has_more"] is False
    assert pagination["cursor"] is None


async def test_non_paginated_listing_unchanged(jp_fetch, listing_dir):
    """不带分页参数时保持原有行为:一次返回全部条目,无 pagination 键。"""
    model = await fetch_page(jp_fetch, listing_dir)
    assert "pagination" not in model
    assert len(model["content"]) == 26
    assert model["format"] == "json"


async def test_sort_by_size_and_last_modified(jp_fetch, listing_dir):
    for sort in ("size", "last_modified"):
        pages, names = await walk_listing(jp_fetch, listing_dir, page_size=10, sort=sort)
        assert sorted(names) == expected_names()
        assert pages[0]["pagination"]["sort"] == sort


# ---------------------------------------------------------------------------
# 并发变化识别(快照水位与连续性)
# ---------------------------------------------------------------------------


async def test_additions_stay_beyond_the_watermark(jp_fetch, jp_root_dir, listing_dir):
    first = await fetch_page(jp_fetch, listing_dir, page_size=10)
    # 快照之后新增一个按名称会排在最前的文件
    (jp_root_dir / listing_dir / "aaa_new.txt").write_text("new", encoding="utf-8")

    rest_names = []
    cursor = first["pagination"]["cursor"]
    seen_changes = []
    while cursor is not None:
        model = await fetch_page(jp_fetch, listing_dir, cursor=cursor)
        rest_names.extend(e["name"] for e in model["content"])
        seen_changes.append(model["pagination"]["changes"])
        cursor = model["pagination"]["cursor"]

    # 新增条目位于可见水位之外:不进入任何后续页面,但被明确通报
    assert "aaa_new.txt" not in rest_names
    assert seen_changes[0]["added"] == ["aaa_new.txt"]
    assert seen_changes[0]["added_count"] == 1
    assert seen_changes[0]["continuity"] == "clean"  # 新增不破坏连续性
    # 已快照的条目不受新增影响:无重复、无遗漏
    assert [e["name"] for e in first["content"]] + rest_names == expected_names()


async def test_deletions_are_reported_and_pages_do_not_shift(jp_fetch, jp_root_dir, listing_dir):
    first = await fetch_page(jp_fetch, listing_dir, page_size=10)
    os.remove(jp_root_dir / listing_dir / "file05.txt")  # 已在首页出现
    os.remove(jp_root_dir / listing_dir / "file15.txt")  # 尚未出现

    second = await fetch_page(jp_fetch, listing_dir, cursor=first["pagination"]["cursor"])
    changes = second["pagination"]["changes"]
    assert changes["continuity"] == "dirty"
    assert changes["deleted_count"] == 2
    assert set(changes["deleted"]) == {"file05.txt", "file15.txt"}
    # 页面仍按快照切片:删除导致短页,而不是后续条目错位
    names = [e["name"] for e in second["content"]]
    assert "file15.txt" not in names
    assert len(names) == 9


async def test_rename_is_reported_as_delete_and_add(jp_fetch, jp_root_dir, listing_dir):
    first = await fetch_page(jp_fetch, listing_dir, page_size=10)
    os.rename(
        jp_root_dir / listing_dir / "file03.txt",
        jp_root_dir / listing_dir / "renamed.txt",
    )
    second = await fetch_page(jp_fetch, listing_dir, cursor=first["pagination"]["cursor"])
    changes = second["pagination"]["changes"]
    assert changes["deleted"] == ["file03.txt"]
    assert changes["added"] == ["renamed.txt"]
    assert changes["continuity"] == "dirty"


async def test_permission_tightening_hides_entries_immediately(jp_fetch, jp_serverapp, listing_dir):
    first = await fetch_page(jp_fetch, listing_dir, page_size=10)
    # 翻页中途收紧过滤策略:file10-file19 立即不可见
    cm = jp_serverapp.contents_manager
    cm.hide_globs = [*cm.hide_globs, "file1*.txt"]

    second = await fetch_page(jp_fetch, listing_dir, cursor=first["pagination"]["cursor"])
    names = [e["name"] for e in second["content"]]
    assert "file10.txt" not in names
    assert "file19.txt" not in names
    changes = second["pagination"]["changes"]
    assert changes["hidden_count"] == 10
    assert "file15.txt" in changes["hidden"]
    assert changes["continuity"] == "dirty"


async def test_loosening_does_not_extend_the_watermark(
    jp_fetch, jp_serverapp, jp_root_dir, listing_dir
):
    (jp_root_dir / listing_dir / ".secret").write_text("hidden", encoding="utf-8")
    first = await fetch_page(jp_fetch, listing_dir, page_size=10)
    assert ".secret" not in [e["name"] for e in first["content"]]

    # 放宽权限后,快照水位不变:.secret 只被通报为 added,不进入页面
    jp_serverapp.contents_manager.allow_hidden = True
    second = await fetch_page(jp_fetch, listing_dir, cursor=first["pagination"]["cursor"])
    assert ".secret" not in [e["name"] for e in second["content"]]
    assert second["pagination"]["changes"]["added"] == [".secret"]


# ---------------------------------------------------------------------------
# 哈希与昂贵元数据的预算
# ---------------------------------------------------------------------------


async def test_hash_budget_defers_expensive_metadata(jp_fetch, listing_dir):
    model = await fetch_page(jp_fetch, listing_dir, page_size=5, hash=1, hash_budget=2)
    entries = model["content"]
    assert len(entries) == 5
    hashes = [e["hash"] for e in entries]
    assert hashes[0] and hashes[1]
    assert hashes[2:] == [None, None, None]
    assert model["pagination"]["hash"] == {
        "requested": True,
        "budget": 2,
        "computed": 2,
        "deferred": 3,
    }


async def test_hash_without_budget_computes_all(jp_fetch, listing_dir):
    model = await fetch_page(jp_fetch, listing_dir, page_size=5, hash=1)
    assert all(e["hash"] for e in model["content"])
    info = model["pagination"]["hash"]
    assert info["computed"] == 5
    assert info["deferred"] == 0
    assert info["budget"] is None


# ---------------------------------------------------------------------------
# 游标生命周期:过期、重启、重试、驱逐、篡改
# ---------------------------------------------------------------------------


async def test_same_page_retry_is_idempotent(jp_fetch, listing_dir):
    first = await fetch_page(jp_fetch, listing_dir, page_size=10)
    cursor = first["pagination"]["cursor"]
    retry_a = await fetch_page(jp_fetch, listing_dir, cursor=cursor)
    retry_b = await fetch_page(jp_fetch, listing_dir, cursor=cursor)
    assert [e["name"] for e in retry_a["content"]] == [e["name"] for e in retry_b["content"]]
    assert retry_a["pagination"]["page_index"] == 1
    assert retry_a["pagination"]["cursor"] == retry_b["pagination"]["cursor"]


async def test_expired_cursor_returns_410(jp_fetch, jp_serverapp, listing_dir):
    jp_serverapp.contents_manager.listing_cursor_ttl = 0
    first = await fetch_page(jp_fetch, listing_dir, page_size=10)
    code, body = await fetch_error(jp_fetch, listing_dir, cursor=first["pagination"]["cursor"])
    assert code == 410
    assert body["reason"] == "cursor_expired"


async def test_server_restart_returns_410(jp_fetch, listing_dir, monkeypatch):
    first = await fetch_page(jp_fetch, listing_dir, page_size=10)
    # 模拟服务重启:进程启动纪元改变,内存中的快照同时丢失
    monkeypatch.setattr(listing, "BOOT_ID", "some-other-boot-epoch")
    code, body = await fetch_error(jp_fetch, listing_dir, cursor=first["pagination"]["cursor"])
    assert code == 410
    assert body["reason"] == "server_restarted"


async def test_unknown_cursor_returns_410(jp_fetch, listing_dir):
    cursor = f"v1.{listing.BOOT_ID}.{'0' * 32}.0"
    code, body = await fetch_error(jp_fetch, listing_dir, cursor=cursor)
    assert code == 410
    assert body["reason"] == "cursor_expired"


async def test_evicted_cursor_returns_410(jp_fetch, jp_serverapp, jp_root_dir, listing_dir):
    other = jp_root_dir / "other_dir"
    other.mkdir()
    (other / "f.txt").write_text("x", encoding="utf-8")
    jp_serverapp.contents_manager.listing_max_cursors = 1
    first = await fetch_page(jp_fetch, listing_dir, page_size=10)
    # 第二个快照驱逐第一个(LRU 上限为 1)
    await fetch_page(jp_fetch, "other_dir", page_size=10)
    code, body = await fetch_error(jp_fetch, listing_dir, cursor=first["pagination"]["cursor"])
    assert code == 410
    assert body["reason"] == "cursor_expired"


async def test_malformed_cursor_returns_400(jp_fetch, listing_dir):
    bad_tokens = [
        "not-a-cursor",
        "",
        "v2." + "x" * 32 + "." + "y" * 32 + ".0",  # 版本不符
        f"v1.{listing.BOOT_ID}.{'0' * 32}.NaN",  # 页码非整数
    ]
    for bad in bad_tokens:
        code, body = await fetch_error(jp_fetch, listing_dir, cursor=bad)
        assert code == 400
        assert body["reason"] == "cursor_malformed"


async def test_foreign_boot_cursor_returns_410(jp_fetch, listing_dir):
    # 结构合法但启动纪元陌生的令牌:按"服务已重启"处理
    cursor = "v1." + "f" * 32 + "." + "0" * 32 + ".0"
    code, body = await fetch_error(jp_fetch, listing_dir, cursor=cursor)
    assert code == 410
    assert body["reason"] == "server_restarted"


async def test_snapshot_too_large_returns_413(jp_fetch, jp_serverapp, listing_dir):
    jp_serverapp.contents_manager.listing_max_snapshot_entries = 3
    code, body = await fetch_error(jp_fetch, listing_dir, page_size=2)
    assert code == 413
    assert body["reason"] == "snapshot_too_large"


# ---------------------------------------------------------------------------
# 请求校验
# ---------------------------------------------------------------------------


async def test_sort_is_fixed_by_the_first_page(jp_fetch, listing_dir):
    first = await fetch_page(jp_fetch, listing_dir, page_size=10, sort="size")
    code, body = await fetch_error(
        jp_fetch, listing_dir, cursor=first["pagination"]["cursor"], sort="name"
    )
    assert code == 400
    assert body["reason"] == "cursor_mismatch"


async def test_page_size_mismatch_returns_400(jp_fetch, listing_dir):
    first = await fetch_page(jp_fetch, listing_dir, page_size=10)
    code, body = await fetch_error(
        jp_fetch, listing_dir, cursor=first["pagination"]["cursor"], page_size=20
    )
    assert code == 400
    assert body["reason"] == "cursor_mismatch"


async def test_invalid_page_size_returns_400(jp_fetch, listing_dir):
    for bad in ("0", "-3", "abc", "1.5", "10000000"):
        code, _ = await fetch_error(jp_fetch, listing_dir, page_size=bad)
        assert code == 400


async def test_invalid_sort_returns_400(jp_fetch, listing_dir):
    code, _ = await fetch_error(jp_fetch, listing_dir, page_size=5, sort="bogus")
    assert code == 400


async def test_sort_and_hash_budget_require_pagination(jp_fetch, listing_dir):
    code, _ = await fetch_error(jp_fetch, listing_dir, sort="name")
    assert code == 400
    code, _ = await fetch_error(jp_fetch, listing_dir, hash_budget=2)
    assert code == 400


async def test_content_zero_with_page_size_returns_400(jp_fetch, listing_dir):
    code, _ = await fetch_error(jp_fetch, listing_dir, page_size=5, content=0)
    assert code == 400


async def test_paginating_a_file_returns_404(jp_fetch, listing_dir):
    code, _ = await fetch_error(jp_fetch, f"{listing_dir}/file00.txt", page_size=5)
    assert code == 404


# ---------------------------------------------------------------------------
# 游标仓库与变化识别的单元测试
# ---------------------------------------------------------------------------


def test_store_token_roundtrip():
    store = ListingCursorStore(ttl=10)
    snapshot = store.create(path="d", names=["a", "b"], sort="name", page_size=1)
    token = store.make_token(snapshot, 1)
    resolved, page_index = store.resolve(token, path="d", sort="name")
    assert resolved is snapshot
    assert page_index == 1


def test_store_ttl_expiry_with_injected_clock():
    now = [100.0]
    store = ListingCursorStore(ttl=10, clock=lambda: now[0])
    snapshot = store.create(path="d", names=["a"], sort="name", page_size=1)
    token = store.make_token(snapshot, 0)
    store.resolve(token, path="d", sort="name")
    now[0] += 11
    with pytest.raises(ListingCursorError) as exc_info:
        store.resolve(token, path="d", sort="name")
    assert exc_info.value.reason == "cursor_expired"
    assert exc_info.value.status_code == 410


def test_store_sliding_expiration():
    now = [100.0]
    store = ListingCursorStore(ttl=10, clock=lambda: now[0])
    snapshot = store.create(path="d", names=["a"], sort="name", page_size=1)
    token = store.make_token(snapshot, 0)
    now[0] += 8
    store.resolve(token, path="d", sort="name")  # 刷新过期时间
    now[0] += 8  # 距上次使用 8 秒,仍在 TTL 内
    resolved, _ = store.resolve(token, path="d", sort="name")
    assert resolved is snapshot


def test_store_lru_eviction_order():
    store = ListingCursorStore(ttl=100, max_cursors=2)
    first = store.create(path="a", names=["x"], sort="name", page_size=1)
    second = store.create(path="b", names=["x"], sort="name", page_size=1)
    # 触碰 first,使 second 成为最久未使用
    store.resolve(store.make_token(first, 0), path="a", sort="name")
    store.create(path="c", names=["x"], sort="name", page_size=1)
    with pytest.raises(ListingCursorError) as exc_info:
        store.resolve(store.make_token(second, 0), path="b", sort="name")
    assert exc_info.value.reason == "cursor_expired"
    resolved, _ = store.resolve(store.make_token(first, 0), path="a", sort="name")
    assert resolved is first


def test_store_snapshot_too_large():
    store = ListingCursorStore(max_entries=2)
    with pytest.raises(ListingCursorError) as exc_info:
        store.create(path="d", names=["a", "b", "c"], sort="name", page_size=1)
    assert exc_info.value.reason == "snapshot_too_large"
    assert exc_info.value.status_code == 413


def test_store_rejects_out_of_range_page():
    store = ListingCursorStore()
    snapshot = store.create(path="d", names=["a", "b"], sort="name", page_size=1)
    token = store.make_token(snapshot, 5)
    with pytest.raises(ListingCursorError) as exc_info:
        store.resolve(token, path="d", sort="name")
    assert exc_info.value.reason == "cursor_mismatch"
    assert exc_info.value.status_code == 400


def test_store_rejects_mismatched_request():
    store = ListingCursorStore()
    snapshot = store.create(path="d", names=["a"], sort="name", page_size=1)
    token = store.make_token(snapshot, 0)
    for kwargs in (
        {"path": "other", "sort": "name"},
        {"path": "d", "sort": "size"},
        {"path": "d", "sort": "name", "page_size": 9},
    ):
        with pytest.raises(ListingCursorError) as exc_info:
            store.resolve(token, **kwargs)
        assert exc_info.value.reason == "cursor_mismatch"


def test_compute_changes_classifies_and_truncates():
    snapshot = [f"file{i:03d}" for i in range(200)] + ["keep1", "now_hidden"]
    visible = ["keep1", "brand_new"]
    raw = ["keep1", "brand_new", "now_hidden"]
    changes = compute_changes(snapshot, visible, raw, max_reported=100)
    assert changes["deleted_count"] == 200
    assert len(changes["deleted"]) == 100
    assert changes["truncated"] is True
    assert changes["hidden"] == ["now_hidden"]
    assert changes["added"] == ["brand_new"]
    assert changes["continuity"] == "dirty"


def test_compute_changes_clean_when_nothing_changed():
    names = ["a", "b", "c"]
    changes = compute_changes(names, names, names)
    assert changes["continuity"] == "clean"
    assert changes["deleted_count"] == changes["added_count"] == changes["hidden_count"] == 0
    assert changes["truncated"] is False
