from src.libs.utilities import add


def test_sum_of_array() -> None:
    arr = [1, 2, 3, 4, 5]
    assert 15 == add(arr)


def test_sum_of_none() -> None:
    assert 0 == add([])
    assert 0 == add(None)
