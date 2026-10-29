import io
import os
import zipfile

import pytest

from faultline.remote_zip import index, read_member


def make_zip(members, force_zip64=False, compression=zipfile.ZIP_DEFLATED):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compression, allowZip64=True) as z:
        for name, data in members.items():
            with z.open(zipfile.ZipInfo(name), "w", force_zip64=force_zip64) as f:
                f.write(data)
    return buf.getvalue()


def fetcher(blob, log):
    def fetch(start, end):
        log.append((start, end))
        return blob[start:end + 1]
    return fetch


@pytest.mark.parametrize("zip64", [False, True])
@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_reads_selected_members(zip64, compression):
    members = {f"images/{i}.jpg": os.urandom(5000 + i) for i in range(200)}
    blob = make_zip(members, force_zip64=zip64, compression=compression)
    log = []
    fetch = fetcher(blob, log)
    infos = {i.filename: i for i in index(fetch, len(blob))}
    assert set(infos) == set(members)
    for name in ("images/3.jpg", "images/150.jpg"):
        assert read_member(fetch, infos[name]) == members[name]


def test_fetches_far_less_than_the_archive():
    members = {f"images/{i}.jpg": os.urandom(20000) for i in range(2000)}
    blob = make_zip(members, compression=zipfile.ZIP_STORED)
    log = []
    fetch = fetcher(blob, log)
    infos = {i.filename: i for i in index(fetch, len(blob))}
    log.clear()
    for name in ("images/10.jpg", "images/1999.jpg"):
        read_member(fetch, infos[name])
    assert len(log) == 2                              # one request per member
    assert sum(e - s + 1 for s, e in log) < 50_000


def test_corruption_is_detected():
    members = {"a.jpg": b"x" * 4000}
    blob = bytearray(make_zip(members, compression=zipfile.ZIP_STORED))
    fetch = fetcher(bytes(blob), [])
    info = index(fetch, len(blob))[0]
    data_at = info.header_offset + 30 + len("a.jpg")
    blob[data_at + 10] ^= 0xFF
    with pytest.raises(IOError, match="CRC"):
        read_member(fetcher(bytes(blob), []), info)
