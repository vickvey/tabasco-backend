from fastapi.responses import JSONResponse
from typing import Optional, Any

class ApiResponse:

    @staticmethod
    def send(
        success: bool,
        status_code: int,
        message: str,
        data: Optional[Any] = None,
    ) -> JSONResponse:
        """
        Helper method to standardize the response structure.
        """
        response_content = {
            "success": success,
            "message": message,
            "data": data,
        }
        return JSONResponse(content=response_content, status_code=status_code)

    @staticmethod
    def success(
        message: str,
        status_code: int = 200,
        data: Optional[Any] = None,
    ) -> JSONResponse:
        """
        Standard success response.
        """
        print(f"[RESPONSE]: Success - {status_code} - {message}")
        return ApiResponse.send(True, status_code, message, data)

    @staticmethod
    def error(
        message: str,
        status_code: int = 400,
        error_data: Optional[Any] = None,
    ) -> JSONResponse:
        """
        Standard error response.
        """
        print(f"[RESPONSE]: Error - {status_code} - {message}")
        return ApiResponse.send(False, status_code, message, error_data)
