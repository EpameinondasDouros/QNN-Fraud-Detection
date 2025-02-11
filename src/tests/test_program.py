import os

from src.program import run

os.environ["DEBUG"] = "true"


def test_should_return_sum() -> None:
    response = run({"values": [10, 5, 20, 7]}, {})
    assert response.result["sum"] == 42
    assert response.metadata is None
