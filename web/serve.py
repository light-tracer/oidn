#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Tiny static file server that disables caching, so rebuilt wasm/js are always
# reloaded during development. Usage: python serve.py [port] [--directory DIR]

import argparse
import http.server
import socketserver

parser = argparse.ArgumentParser()
parser.add_argument("port", nargs="?", type=int, default=8778)
parser.add_argument("--directory", default=".")
args = parser.parse_args()


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


class Handler(NoCacheHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=args.directory, **kw)


with socketserver.ThreadingTCPServer(("127.0.0.1", args.port), Handler) as httpd:
    print(f"serving {args.directory} at http://127.0.0.1:{args.port} (no-cache)")
    httpd.serve_forever()
