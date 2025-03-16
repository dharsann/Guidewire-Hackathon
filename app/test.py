import unittest
from unittest.mock import patch
from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

class TestSelfHealAPI(unittest.TestCase):
    @patch('your_file.run_kubectl')
    def test_pod_crash_remediation(self, mock_kubectl):
        mock_kubectl.return_value = "pod1 pod2"
        test_data = {
            "input_features": [0.8, 0.8, 0.3],
            "pod_lifetime_seconds": 30,  
            "pod_name": "test-pod" 
        }
        response = client.post("/self_heal", json=test_data)
        result = response.json()
        self.assertEqual(result["issue_type"], "pod_crash")
        self.assertIn("Restarted pod", result["action"])
        mock_kubectl.assert_called_with(["kubectl", "delete", "pod", "test-pod"])

if __name__ == "__main__":
    unittest.main()