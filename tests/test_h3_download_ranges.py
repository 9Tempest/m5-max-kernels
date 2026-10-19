"""Mock HTTP and CPU filesystem tests only; no requests leave this process."""

import hashlib
import importlib.util
import io
from pathlib import Path
import re
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest


MODULE = Path(__file__).resolve().parents[1] / "integrations/h3/download_ranges.py"
spec = importlib.util.spec_from_file_location("h3_download_ranges", MODULE)
ranges = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ranges)


class Response(io.BytesIO):
    def __init__(self, data, status=206, **headers):
        super().__init__(data)
        self.status = status
        self.headers = headers


class Server:
    def __init__(self, data):
        self.data = data
        self.requests = []

    def __call__(self, request, timeout):
        assert timeout > 0
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        start, end = map(int, match.groups())
        self.requests.append((start, end))
        return Response(
            self.data[start : end + 1],
            **{
                "Content-Range": f"bytes {start}-{end}/{len(self.data)}",
                "Content-Length": str(end - start + 1),
            },
        )


def entry(data):
    return dict(
        path="FL2VA/video_vae/source/model.safetensors",
        size=len(data),
        lfs=dict(oid=hashlib.sha256(data).hexdigest()),
    )


def prefix_file(target, data):
    path = target / "FL2VA/video_vae/source/model.safetensors.partial"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return path


def no_processes(_):
    pass


def test_completes_only_tail_preserving_prefix_and_pinned_digest(tmp_path):
    data = bytes(range(100))
    prefix = prefix_file(tmp_path, data[:23])
    before = prefix.stat().st_mtime_ns
    server = Server(data)
    result = ranges.download_one(
        entry(data), tmp_path, workers=3, chunk_bytes=16, opener=server, guard=no_processes
    )
    assert result["status"] == "verified"
    assert prefix.read_bytes() == data[:23]
    assert prefix.stat().st_mtime_ns == before
    assert min(start for start, _ in server.requests) == 23
    assert (tmp_path / entry(data)["path"]).read_bytes() == data
    assert sum(end - start + 1 for start, end in server.requests) == 77


def test_truncated_chunk_resumes_from_its_existing_bytes(tmp_path):
    data = b"abcdefghij"
    path = tmp_path / "chunk.part"

    def truncated(request, timeout):
        return Response(b"ef", **{"Content-Range": "bytes 4-9/10", "Content-Length": "6"})

    with pytest.raises(ranges.DownloadError, match="truncated"):
        ranges.fetch_part("https://example.invalid/file", path, 4, 9, 10, truncated)
    assert path.read_bytes() == b"ef"
    server = Server(data)
    result = ranges.fetch_part("https://example.invalid/file", path, 4, 9, 10, server)
    assert server.requests == [(6, 9)]
    assert result["resumed_bytes"] == 2
    assert path.read_bytes() == b"efghij"


@pytest.mark.parametrize(
    "status,headers,body,message",
    [
        (200, {"Content-Range": "bytes 0-2/3"}, b"abc", "HTTP 206"),
        (206, {"Content-Range": "bytes 1-2/3"}, b"abc", "Content-Range"),
        (206, {"Content-Range": "bytes 0-2/4"}, b"abc", "Content-Range"),
        (206, {"Content-Range": "bytes 0-2/*"}, b"abc", "Content-Range"),
        (206, {"Content-Range": "bytes 0-2/3", "Content-Length": "4"}, b"abc", "Content-Length"),
        (206, {"Content-Range": "bytes 0-2/3", "Content-Encoding": "gzip"}, b"abc", "compressed"),
    ],
)
def test_rejects_invalid_http_contract_before_writing(tmp_path, status, headers, body, message):
    path = tmp_path / "chunk.part"
    with pytest.raises(ranges.DownloadError, match=message):
        ranges.fetch_part(
            "https://example.invalid/file",
            path,
            0,
            2,
            3,
            lambda *_args, **_kwargs: Response(body, status, **headers),
        )
    assert not path.exists()


def test_excess_body_is_not_accepted_as_verified_on_resume(tmp_path):
    path = tmp_path / "chunk.part"
    with pytest.raises(ranges.DownloadError, match="exceeded"):
        ranges.fetch_part(
            "https://example.invalid/file",
            path,
            0,
            2,
            3,
            lambda *_args, **_kwargs: Response(
                b"abcd", **{"Content-Range": "bytes 0-2/3", "Content-Length": "3"}
            ),
        )
    server = Server(b"abc")
    ranges.fetch_part("https://example.invalid/file", path, 0, 2, 3, server)
    assert server.requests == [(0, 2)]
    assert len(list(tmp_path.glob("*.unverified-*"))) == 1


def test_final_sha_mismatch_retains_original_and_all_chunks(tmp_path):
    good = b"0123456789"
    prefix = prefix_file(tmp_path, good[:3])
    result = ranges.download_one(
        entry(good), tmp_path, chunk_bytes=3, opener=Server(b"012xxxxxxx"), guard=no_processes
    )
    assert result["status"] == "failed"
    assert "SHA256" in result["error"]
    assert prefix.read_bytes() == good[:3]
    assert not (tmp_path / entry(good)["path"]).exists()
    assert list(prefix.parent.glob("*.ranges/*.part"))
    assert list(prefix.parent.glob("*.ranges/assembly-*.partial"))


def test_failed_download_resumes_completed_chunks_without_refetch(tmp_path):
    data = b"0123456789"
    prefix_file(tmp_path, data[:2])
    server = Server(data)

    def flaky(request, timeout):
        if request.get_header("Range") == "bytes=6-9":
            raise HTTPError("https://signed.invalid/?secret=hidden", 503, "hidden", {}, None)
        return server(request, timeout)

    first = ranges.download_one(
        entry(data), tmp_path, chunk_bytes=4, opener=flaky, guard=no_processes
    )
    assert first["status"] == "failed"
    second_server = Server(data)
    second = ranges.download_one(
        entry(data), tmp_path, chunk_bytes=4, opener=second_server, guard=no_processes
    )
    assert second["status"] == "verified"
    assert second_server.requests == [(6, 9)]
    attempts = list(tmp_path.rglob("attempt-*.json"))
    assert len(attempts) == 2
    assert "secret" not in "".join(path.read_text() for path in attempts)


def test_shared_download_lock_blocks_before_network(tmp_path):
    def no_network(*_args, **_kwargs):
        pytest.fail("must not open a network request while another downloader owns the lock")

    with ranges.model_lock(tmp_path):
        with pytest.raises(ranges.DownloadError, match="another downloader"):
            ranges.download_one(entry(b"test"), tmp_path, opener=no_network, guard=no_processes)


def test_legacy_downloader_without_lock_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ranges.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="123 python integrations/h3/download.py --workers 16\n", stderr=""
        ),
    )
    with pytest.raises(ranges.DownloadError, match="legacy H3 downloader"):
        ranges.assert_no_legacy_writer(tmp_path / "file.partial")


def test_orphaned_prefix_writer_is_refused(tmp_path, monkeypatch):
    prefix = tmp_path / "file.partial"
    prefix.write_bytes(b"data")
    responses = iter(
        [
            SimpleNamespace(
                returncode=0, stdout="123 curl https://signed.invalid/?secret=hidden\n", stderr=""
            ),
            SimpleNamespace(returncode=0, stdout="123\n", stderr=""),
        ]
    )
    monkeypatch.setattr(ranges.subprocess, "run", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(ranges.shutil, "which", lambda _: "lsof")
    with pytest.raises(ranges.DownloadError, match="open in another process"):
        ranges.assert_no_legacy_writer(prefix)


def test_changing_prefix_aborts_assembly(tmp_path):
    data = b"0123456789"
    prefix = prefix_file(tmp_path, data[:2])
    server = Server(data)

    def mutate(request, timeout):
        prefix.write_bytes(b"changed")
        return server(request, timeout)

    result = ranges.download_one(
        entry(data), tmp_path, chunk_bytes=8, opener=mutate, guard=no_processes
    )
    assert result["status"] == "failed"
    assert "prefix changed" in result["error"]
    assert not (tmp_path / entry(data)["path"]).exists()


def test_refuses_manifest_replacement_and_unsafe_path(tmp_path):
    tree = tmp_path / "tree.json"
    tree.write_text("[]")
    with pytest.raises(ranges.DownloadError, match="pinned H3 snapshot"):
        ranges.selected_entry(tree, "file.safetensors")
    bad = entry(b"data") | {"path": "../file.safetensors"}
    with pytest.raises(ranges.DownloadError, match="safe relative"):
        ranges.validate_entry(bad)


def test_unexpected_transport_errors_are_sanitized():
    error = OSError("https://signed.invalid/?secret=hidden")
    assert "secret" not in ranges.safe_error(error)
    assert "signed.invalid" not in ranges.safe_error(error)
