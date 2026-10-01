"""Unit tests for Unreal Bridge HTTP endpoint."""

import unittest
import json
import numpy as np
from io import BytesIO
from unittest.mock import patch, MagicMock
from pathlib import Path

# Mock SAM2 before importing the app
import sys
sys.modules["sam2"] = MagicMock()
sys.modules["sam2.build_sam"] = MagicMock()
sys.modules["sam2.automatic_mask_generator"] = MagicMock()

# Now we can import the app
import run_unreal_bridge


class TestUnrealBridge(unittest.TestCase):
    def setUp(self):
        """Set up Flask test client."""
        run_unreal_bridge.app.testing = True
        self.client = run_unreal_bridge.app.test_client()

    def test_health_endpoint(self):
        """Test /health returns ok."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["status"], "ok")

    def test_process_frame_empty_body(self):
        """Test /process_frame with empty body returns 400."""
        resp = self.client.post("/process_frame", data=b"")
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.data)
        self.assertIn("error", data)

    @patch("run_unreal_bridge.get_sam2_predictor")
    def test_process_frame_with_image(self, mock_get_predictor):
        """Test /process_frame with dummy JPEG image."""
        # Create a simple dummy image (1x1 RGB)
        try:
            import cv2
        except ImportError:
            self.skipTest("cv2 not available")
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, img_bytes = cv2.imencode(".jpg", img)
        
        # Mock SAM2
        mock_predictor = MagicMock()
        mock_get_predictor.return_value = mock_predictor
        
        resp = self.client.post("/process_frame", data=img_bytes.tobytes(), content_type="image/jpeg")
        
        # Should succeed with 200
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertEqual(data["status"], "ok")
        self.assertIn("masks", data)
        self.assertIn("boxes", data)
        self.assertIn("labels", data)
        self.assertIn("time_ms", data)

    @patch("run_unreal_bridge.requests.post")
    def test_forward_results_to_unreal(self, mock_post):
        """Test forwarding results to external server."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp
        
        results = {
            "masks": 5,
            "boxes": [[10, 20, 30, 40]],
            "labels": [0.95],
            "time_ms": 1234.5
        }
        
        success = run_unreal_bridge.forward_results_to_unreal(results)
        
        if run_unreal_bridge.requests and run_unreal_bridge.FORWARD_TO_UNREAL and run_unreal_bridge.SERVER_URL:
            self.assertTrue(mock_post.called)


if __name__ == "__main__":
    unittest.main()
