import unittest
from flow import run_flow
from crewai_tools import FileReadTool

class FlowTest(unittest.TestCase):
    def test_run_flow_returns_string(self):
        result = run_flow("Hello")
        self.assertIsInstance(result, str)

    def test_file_read_tool(self):
        tool = FileReadTool()
        content = tool.run("knowledge/secret.md")
        self.assertIn("Banana", content)

if __name__ == "__main__":
    unittest.main()
