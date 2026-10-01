import os
import json
import unittest
from unittest.mock import patch, MagicMock

from tools import remote_adapter


class TestRemoteAdapter(unittest.TestCase):
    def setUp(self):
        # ensure we don't require real server
        os.environ["SERVER_URL"] = "http://example.com"
        os.environ["ENABLE_REMOTE"] = "true"

    @patch("tools.remote_adapter.requests.Session.post")
    def test_send_image_bytes_rest_multipart(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        mock_post.return_value = mock_resp

        data = b"fakeimagebytes"
        resp = remote_adapter.send_image_bytes_rest(data, filename="f.jpg")
        self.assertEqual(resp, {"ok": True})
        self.assertTrue(mock_post.called)

    @patch("tools.remote_adapter.requests.Session.post")
    def test_send_image_bytes_rest_fallback_json(self, mock_post):
        # first call: raise an error to force fallback
        def side_effect(*args, **kwargs):
            if kwargs.get("files"):
                # simulate failure on multipart
                raise Exception("multipart failed")
            m = MagicMock()
            m.raise_for_status = MagicMock()
            m.json.return_value = {"ok": "fallback"}
            return m

        mock_post.side_effect = side_effect
        data = b"fakeimagebytes"
        resp = remote_adapter.send_image_bytes_rest(data, filename="f.jpg")
        self.assertEqual(resp, {"ok": "fallback"})


if __name__ == "__main__":
    unittest.main()
