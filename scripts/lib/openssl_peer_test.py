#!/usr/bin/env -S uv run python
"""Test the pinned OpenSSL peer's download check and cache without building OpenSSL."""

from pathlib import Path
import tempfile
import unittest

import openssl_peer


class OpenSslPeerTest(unittest.TestCase):
    def test_download_refuses_other_content_and_leaves_no_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'tampered.tar.gz'
            source.write_bytes(b'not the OpenSSL release')
            destination = Path(temporary) / 'download.tar.gz'
            with self.assertRaisesRegex(ValueError, openssl_peer.SHA256):
                openssl_peer.download(destination, url=source.as_uri())
            self.assertFalse(destination.exists())

    def test_cached_binary_is_reused_without_fetching(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = openssl_peer.prefix(root) / 'bin/openssl'
            executable.parent.mkdir(parents=True)
            executable.write_text('#!/bin/sh\n')

            def fetch(_destination):
                raise AssertionError('a cached peer must not be fetched again')

            self.assertEqual(openssl_peer.binary(root, fetch), executable)


if __name__ == '__main__':
    unittest.main()
