"""Tests for paginated directory listings backed by bounded snapshot cursors."""

import asyncio
import json
import os
import warnings

import pytest
import tornado

from ...utils import expected_http_error


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
def page_dir(contents_dir):
    """A directory with a controlled set of entries for pagination tests."""
    d = contents_dir / "paginate"
    d.mkdir()
    for i in range(25):
        d.joinpath(f"file{i:02d}.txt").write_text(f"content {i}", encoding="utf-8")
    d.joinpath("subdir").mkdir()
    d.joinpath(".hidden").write_text("secret", encoding="utf-8")
    return d


async def fetch_page(jp_fetch, path, **params):
    """GET one page of a directory listing and return the decoded model."""
    response = await jp_fetch("api", "contents", path, method="GET", params=params)
    return json.loads(response.body.decode())


async def walk_pages(jp_fetch, path, **params):
    """Fetch every page of a listing; return (entries, page_models)."""
    entries = []
    pages = []
    cursor = None
    while True:
        page_params = dict(params)
        if cursor is not None:
            page_params["cursor"] = cursor
        model = await fetch_page(jp_fetch, path, **page_params)
        pages.append(model)
        entries.extend(model["content"])
        cursor = model["page"]["cursor"]
        if cursor is None:
            break
    return entries, pages


async def test_first_page_freezes_sort_and_watermark(jp_fetch, page_dir):
    model = await fetch_page(jp_fetch, "paginate", page_size="10")
    page = model["page"]
    assert model["type"] == "directory"
    assert model["format"] == "json"
    names = [e["name"] for e in model["content"]]
    assert names == sorted(names)
    assert len(names) == 10
    assert page["has_more"] is True
    assert page["cursor"]
    assert page["sort"] == "name"
    assert page["sort_dir"] == "asc"
    assert page["returned"] == 10
    assert page["skipped"] == 0
    assert page["hashes_deferred"] is False
    # The first page fixes the visible watermark and reports no changes.
    assert page["changes"] is None
    watermark = page["snapshot"]["watermark"]
    assert watermark["entry_count"] == 27  # 25 files + subdir + .hidden
    assert page["snapshot"]["entries"] == 27
    assert page["snapshot"]["created"]
    assert page["snapshot"]["expires_at"]
    # Hidden files stay out of the listing.
    assert ".hidden" not in names


async def test_walk_all_pages_matches_unpaginated(jp_fetch, page_dir):
    entries, pages = await walk_pages(jp_fetch, "paginate", page_size="7")
    assert len(pages) == 4  # 26 visible entries, 7 per page
    names = [e["name"] for e in entries]
    assert len(names) == len(set(names))  # no duplicates
    assert ".hidden" not in names

    plain = await fetch_page(jp_fetch, "paginate")
    assert "page" not in plain  # non-paginated responses are unchanged
    plain_names = sorted(e["name"] for e in plain["content"])
    assert names == plain_names  # same set, stable order, nothing omitted


async def test_non_paginated_calls_stay_compatible(jp_fetch, page_dir):
    model = await fetch_page(jp_fetch, "paginate")
    assert "page" not in model
    assert len(model["content"]) == 26
    # content=0 is untouched as well
    model = await fetch_page(jp_fetch, "paginate", content="0")
    assert model["content"] is None
    assert "page" not in model


async def test_concurrent_add_is_reported_but_not_served(jp_fetch, page_dir):
    model = await fetch_page(jp_fetch, "paginate", page_size="10")
    cursor = model["page"]["cursor"]

    # A new file that would sort into the already-served region.
    page_dir.joinpath("aaa_new.txt").write_text("new", encoding="utf-8")

    seen = []
    changes = None
    while cursor is not None:
        model = await fetch_page(jp_fetch, "paginate", cursor=cursor)
        seen.extend(e["name"] for e in model["content"])
        changes = model["page"]["changes"]
        cursor = model["page"]["cursor"]

    # The snapshot keeps the new file out of every page...
    assert "aaa_new.txt" not in seen
    # ...but the change summary identifies it, and flags that the pages
    # already served no longer form a consistent prefix of the directory.
    assert "aaa_new.txt" in changes["added"]
    assert changes["added_count"] == 1
    assert changes["continuity_broken"] is True


async def test_concurrent_delete_of_served_entry_breaks_continuity(jp_fetch, page_dir):
    model = await fetch_page(jp_fetch, "paginate", page_size="10")
    cursor = model["page"]["cursor"]
    served = [e["name"] for e in model["content"]]

    os.remove(page_dir / served[0])

    model = await fetch_page(jp_fetch, "paginate", cursor=cursor)
    changes = model["page"]["changes"]
    assert served[0] in changes["removed"]
    assert changes["removed_count"] == 1
    assert changes["continuity_broken"] is True


async def test_concurrent_delete_of_unserved_entry_is_skipped(jp_fetch, page_dir):
    model = await fetch_page(jp_fetch, "paginate", page_size="10")
    cursor = model["page"]["cursor"]

    # file20.txt sorts past the first page, so it has not been served yet.
    os.remove(page_dir / "file20.txt")

    seen = []
    last_changes = None
    while cursor is not None:
        model = await fetch_page(jp_fetch, "paginate", cursor=cursor)
        seen.extend(e["name"] for e in model["content"])
        last_changes = model["page"]["changes"]
        cursor = model["page"]["cursor"]

    # The deleted entry is skipped when its page comes up, and the summary
    # reports the removal without breaking the continuity of served pages.
    assert "file20.txt" not in seen
    assert "file20.txt" in last_changes["removed"]
    assert last_changes["continuity_broken"] is False


async def test_permission_tightening_hides_entries_immediately(jp_fetch, jp_serverapp, page_dir):
    model = await fetch_page(jp_fetch, "paginate", page_size="10")
    cursor = model["page"]["cursor"]

    # Permissions tighten between two pages.
    jp_serverapp.contents_manager.hide_globs = ["file1*.txt"]

    seen = []
    while cursor is not None:
        model = await fetch_page(jp_fetch, "paginate", cursor=cursor)
        seen.extend(e["name"] for e in model["content"])
        cursor = model["page"]["cursor"]

    hidden = [n for n in seen if n.startswith("file1")]
    assert hidden == []
    assert "file20.txt" in seen  # unaffected entries keep flowing


async def test_hidden_file_tightening_via_allow_hidden(jp_fetch, jp_serverapp, contents_dir):
    d = contents_dir / "with_hidden"
    d.mkdir()
    for i in range(6):
        d.joinpath(f".h{i}.txt").write_text("h", encoding="utf-8")
        d.joinpath(f"v{i}.txt").write_text("v", encoding="utf-8")

    cm = jp_serverapp.contents_manager
    cm.allow_hidden = True
    model = await fetch_page(jp_fetch, "with_hidden", page_size="4")
    names = [e["name"] for e in model["content"]]
    assert names == [".h0.txt", ".h1.txt", ".h2.txt", ".h3.txt"]
    cursor = model["page"]["cursor"]
    assert cursor is not None

    # Tightening allow_hidden mid-pagination hides the remaining dotfiles
    # at once: the very next page contains only regular files.
    cm.allow_hidden = False
    model = await fetch_page(jp_fetch, "with_hidden", cursor=cursor)
    names = [e["name"] for e in model["content"]]
    assert names == ["v0.txt", "v1.txt", "v2.txt", "v3.txt"]


async def test_hash_budget_defers_expensive_metadata(jp_fetch, page_dir):
    model = await fetch_page(jp_fetch, "paginate", page_size="5", hash="1", hash_budget="2")
    page = model["page"]
    files = [e for e in model["content"] if e["type"] != "directory"]
    hashed = [e for e in files if e["hash"] is not None]
    deferred = [e for e in files if e["hash"] is None]
    assert len(hashed) == 2
    assert len(deferred) == 3
    assert all(e["hash_algorithm"] for e in hashed)
    assert page["hashes_deferred"] is True

    # A generous budget hashes everything on the page.
    model = await fetch_page(jp_fetch, "paginate", page_size="5", hash="1", hash_budget="5")
    assert model["page"]["hashes_deferred"] is False
    assert all(e["hash"] for e in model["content"] if e["type"] != "directory")


async def test_hash_budget_is_capped_by_config(jp_fetch, jp_serverapp, page_dir):
    jp_serverapp.contents_manager.listing_hash_budget_max = 3
    model = await fetch_page(jp_fetch, "paginate", page_size="5", hash="1", hash_budget="100")
    files = [e for e in model["content"] if e["type"] != "directory"]
    assert len([e for e in files if e["hash"] is not None]) == 3
    assert model["page"]["hashes_deferred"] is True


async def test_cursor_expired_returns_410(jp_fetch, jp_serverapp, page_dir):
    jp_serverapp.contents_manager.listing_cursor_ttl = 0.2
    model = await fetch_page(jp_fetch, "paginate", page_size="5")
    cursor = model["page"]["cursor"]
    await asyncio.sleep(0.4)
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await fetch_page(jp_fetch, "paginate", cursor=cursor)
    assert expected_http_error(e, 410)


async def test_unknown_cursor_returns_410(jp_fetch, page_dir):
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await fetch_page(jp_fetch, "paginate", cursor="dsc1.0123456789abcdef0123456789abcdef.1")
    assert expected_http_error(e, 410)


async def test_cursor_does_not_survive_restart(jp_fetch, jp_serverapp, page_dir):
    model = await fetch_page(jp_fetch, "paginate", page_size="5")
    cursor = model["page"]["cursor"]
    # Simulate a server restart: the process-local cursor store is gone.
    jp_serverapp.contents_manager._listing_cursor_store = None
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await fetch_page(jp_fetch, "paginate", cursor=cursor)
    assert expected_http_error(e, 410)


async def test_same_page_retry_is_idempotent(jp_fetch, page_dir):
    model = await fetch_page(jp_fetch, "paginate", page_size="10")
    cursor = model["page"]["cursor"]

    page2 = await fetch_page(jp_fetch, "paginate", cursor=cursor)
    retry = await fetch_page(jp_fetch, "paginate", cursor=cursor)

    assert [e["name"] for e in retry["content"]] == [e["name"] for e in page2["content"]]
    assert retry["page"]["cursor"] == page2["page"]["cursor"]

    # The retried cursor still leads to the remaining pages.
    page3 = await fetch_page(jp_fetch, "paginate", cursor=page2["page"]["cursor"])
    assert page3["page"]["has_more"] is False
    assert len(page3["content"]) == 6  # 26 visible entries: 10 + 10 + 6


async def test_replay_of_last_page(jp_fetch, page_dir):
    model = await fetch_page(jp_fetch, "paginate", page_size="25")
    cursor = model["page"]["cursor"]
    last = await fetch_page(jp_fetch, "paginate", cursor=cursor)
    assert last["page"]["has_more"] is False
    assert last["page"]["cursor"] is None

    retry = await fetch_page(jp_fetch, "paginate", cursor=cursor)
    assert [e["name"] for e in retry["content"]] == [e["name"] for e in last["content"]]
    assert retry["page"]["cursor"] is None


async def test_consumed_cursor_outside_retry_window_returns_409(jp_fetch, page_dir):
    model = await fetch_page(jp_fetch, "paginate", page_size="5")
    cursor = model["page"]["cursor"]
    session_id = cursor.split(".")[1]
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await fetch_page(jp_fetch, "paginate", cursor=f"dsc1.{session_id}.999")
    assert expected_http_error(e, 409)


async def test_malformed_cursor_returns_400(jp_fetch, page_dir):
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await fetch_page(jp_fetch, "paginate", cursor="not-a-cursor")
    assert expected_http_error(e, 400)


async def test_cursor_from_other_directory_returns_400(jp_fetch, page_dir):
    model = await fetch_page(jp_fetch, "paginate", page_size="5")
    cursor = model["page"]["cursor"]
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await fetch_page(jp_fetch, "", cursor=cursor)
    assert expected_http_error(e, 400)


async def test_sort_is_fixed_by_first_page(jp_fetch, page_dir):
    model = await fetch_page(jp_fetch, "paginate", page_size="5", sort="name")
    cursor = model["page"]["cursor"]
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await fetch_page(jp_fetch, "paginate", cursor=cursor, sort="last_modified")
    assert expected_http_error(e, 400)
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await fetch_page(jp_fetch, "paginate", cursor=cursor, sort_dir="desc")
    assert expected_http_error(e, 400)


async def test_sort_last_modified_desc(jp_fetch, contents_dir):
    d = contents_dir / "by_mtime"
    d.mkdir()
    for i in range(5):
        p = d / f"f{i}.txt"
        p.write_text("x", encoding="utf-8")
        ts = 1_000_000 + i * 100
        os.utime(p, (ts, ts))

    entries, _ = await walk_pages(
        jp_fetch, "by_mtime", page_size="2", sort="last_modified", sort_dir="desc"
    )
    names = [e["name"] for e in entries]
    assert names == ["f4.txt", "f3.txt", "f2.txt", "f1.txt", "f0.txt"]


async def test_page_size_validation(jp_fetch, page_dir):
    for bad in ("0", "-1", "99999", "abc"):
        with pytest.raises(tornado.httpclient.HTTPClientError) as e:
            await fetch_page(jp_fetch, "paginate", page_size=bad)
        assert expected_http_error(e, 400)


async def test_pagination_requires_content(jp_fetch, page_dir):
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await fetch_page(jp_fetch, "paginate", content="0", page_size="5")
    assert expected_http_error(e, 400)


async def test_pagination_rejects_files(jp_fetch, page_dir):
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await fetch_page(jp_fetch, "paginate/file00.txt", page_size="5")
    assert expected_http_error(e, 400)


async def test_pagination_rejects_invalid_sort(jp_fetch, page_dir):
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await fetch_page(jp_fetch, "paginate", page_size="5", sort="size")
    assert expected_http_error(e, 400)


async def test_directory_too_large_returns_413(jp_fetch, jp_serverapp, page_dir):
    jp_serverapp.contents_manager.listing_snapshot_max_entries = 5
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await fetch_page(jp_fetch, "paginate", page_size="5")
    assert expected_http_error(e, 413)


async def test_cursor_store_is_bounded(jp_fetch, jp_serverapp, page_dir):
    jp_serverapp.contents_manager.listing_cursor_max_sessions = 2
    first = (await fetch_page(jp_fetch, "paginate", page_size="5"))["page"]["cursor"]
    await fetch_page(jp_fetch, "paginate", page_size="5")
    await fetch_page(jp_fetch, "paginate", page_size="5")
    # The least recently used session was evicted.
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await fetch_page(jp_fetch, "paginate", cursor=first)
    assert expected_http_error(e, 410)


async def test_empty_directory(jp_fetch, contents_dir):
    (contents_dir / "empty_dir").mkdir()
    model = await fetch_page(jp_fetch, "empty_dir", page_size="10")
    assert model["content"] == []
    assert model["page"]["has_more"] is False
    assert model["page"]["cursor"] is None


async def test_nonexistent_directory_returns_404(jp_fetch, page_dir):
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await fetch_page(jp_fetch, "no_such_dir", page_size="5")
    assert expected_http_error(e, 404)


async def test_deleted_directory_returns_404(jp_fetch, page_dir):
    model = await fetch_page(jp_fetch, "paginate", page_size="5")
    cursor = model["page"]["cursor"]
    for entry in sorted(page_dir.iterdir()):
        if entry.is_dir():
            entry.rmdir()
        else:
            entry.unlink()
    page_dir.rmdir()
    with pytest.raises(tornado.httpclient.HTTPClientError) as e:
        await fetch_page(jp_fetch, "paginate", cursor=cursor)
    assert expected_http_error(e, 404)
