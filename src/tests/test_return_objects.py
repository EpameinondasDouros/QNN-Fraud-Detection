from src.libs.return_objects import ErrorResponse, ResultResponse


def test_error_response() -> None:
    response = ErrorResponse("200", "OK")
    assert response.to_json() == '{"code": "200", "detail": "OK"}'


def test_result_response() -> None:
    response = ResultResponse({"value": 5}, {"lang": "python"})
    assert response.to_json() == '{"result": {"value": 5}, "metadata": {"lang": "python"}}'
