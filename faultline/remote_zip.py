"""Read selected members of a large remote zip file with HTTP range requests.

The zip index (central directory) sits at the end of the file, so the index plus the
wanted members can be fetched without downloading the whole archive. The transport is a
plain function fetch(start, end) -> bytes (end inclusive), which keeps this testable.
"""

import struct
import urllib.request
import zipfile
import zlib


def http_fetcher(url, timeout=60):
    def fetch(start, end):
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 206:
                raise IOError(f"server ignored the range request (HTTP {r.status})")
            return r.read()
    return fetch


def http_size(url, timeout=60):
    """Total size from a one-byte range request; fails clearly if ranges are unsupported."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        rng = r.headers.get("Content-Range", "")
        if r.status != 206 or "/" not in rng:
            raise IOError(f"{url} does not support range requests (HTTP {r.status})")
        return int(rng.rsplit("/", 1)[1])


class RangeFile:
    """Seekable read-only file over fetch(); caches fixed-size blocks."""

    def __init__(self, fetch, size, block=1 << 20):
        self.fetch, self.size, self.block = fetch, size, block
        self.pos = 0
        self.cache = {}

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            start = i * self.block
            end = min(start + self.block, self.size) - 1
            self.cache[i] = self.fetch(start, end)
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n > 4 * self.block:                    # large reads (the index) in one request
            data = self.fetch(self.pos, self.pos + n - 1)
        else:
            out = []
            p, left = self.pos, n
            while left:
                i, off = divmod(p, self.block)
                chunk = self._block(i)[off:off + left]
                out.append(chunk)
                p += len(chunk)
                left -= len(chunk)
            data = b"".join(out)
        self.pos += len(data)
        return data


def index(fetch, size):
    """ZipInfo objects for every member, read from the remote central directory."""
    with zipfile.ZipFile(RangeFile(fetch, size)) as z:
        return z.infolist()


LOCAL_HEADER = struct.Struct("<4s5H3I2H")       # signature ... name length, extra length


def read_member(fetch, info, slack=1024):
    """Fetch one member with a single range request and return its bytes."""
    start = info.header_offset
    head_guess = LOCAL_HEADER.size + len(info.filename.encode()) + slack
    raw = fetch(start, start + head_guess + info.compress_size - 1)
    fields = LOCAL_HEADER.unpack_from(raw)
    if fields[0] != b"PK\x03\x04":
        raise IOError(f"bad local header for {info.filename}")
    name_len, extra_len = fields[-2], fields[-1]
    data_start = LOCAL_HEADER.size + name_len + extra_len
    data = raw[data_start:data_start + info.compress_size]
    if len(data) < info.compress_size:              # extra field longer than the guess
        more = fetch(start + len(raw), start + data_start + info.compress_size - 1)
        data = (raw + more)[data_start:data_start + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        out = data
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        out = zlib.decompress(data, -15)
    else:
        raise IOError(f"unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(out) != info.CRC:
        raise IOError(f"CRC mismatch for {info.filename}")
    return out
