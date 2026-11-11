"""
支持 Range（206 Partial Content）的静态文件服务器，用于从 Featurize 取回大文件。
默认 http.server 在该实例上不处理 Range，这里基于 BaseHTTPRequestHandler 显式实现：
普通 GET 返回 200，带 Range 头返回 206 + Content-Range + Accept-Ranges。
用法：python range_server.py [port]，以当前目录为根。
"""
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)$")


class RangeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _safe_path(self):
        rel = self.path.split("?", 1)[0].split("#", 1)[0]
        rel = rel.lstrip("/")
        root = os.path.abspath(os.getcwd())
        full = os.path.abspath(os.path.join(root, rel))
        if not (full == root or full.startswith(root + os.sep)):
            return None
        return full

    def do_GET(self):
        full = self._safe_path()
        if full is None or not os.path.isfile(full):
            self.send_error(404)
            return
        size = os.path.getsize(full)
        m = RANGE_RE.match(self.headers.get("Range", "").strip())
        start, end = 0, size - 1
        partial = False
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
            end = min(end, size - 1)
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */%d" % size)
                self.end_headers()
                return
            partial = True
        length = end - start + 1
        ctype = "application/octet-stream"
        with open(full, "rb") as f:
            f.seek(start)
            data = f.read(length)
        if partial:
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    HTTPServer(("0.0.0.0", port), RangeHandler).serve_forever()


if __name__ == "__main__":
    main()
