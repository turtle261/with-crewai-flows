import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.flow import agent, task
from crewai import Crew


def test_file_read_tool(tmp_path):
    sample = Path('sample.txt')
    sample.write_text('Revenue was $1000 in Q4.')
    try:
        query = f"Read the file {sample} and repeat its contents."
        task.description = f'User asks: "{query}"'
        result = Crew(agents=[agent], tasks=[task]).kickoff()
        assert 'Revenue was $1000 in Q4.' in result.raw
    finally:
        sample.unlink()
