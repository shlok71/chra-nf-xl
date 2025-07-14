import unittest
import subprocess

class TestOfflineMode(unittest.TestCase):

    def test_offline_mode(self):
        # This is a placeholder test. A real test would check that
        # the application runs correctly in offline mode.
        result = subprocess.run(["python3", "neuroforge_launcher.py", "--offline"], capture_output=True, text=True)
        self.assertIn("Loading NeuroForge with the following configuration:", result.stdout)
        self.assertIn("\"offline\": true", result.stdout)

if __name__ == "__main__":
    unittest.main()
