class EquiReachError(Exception):
    def __init__(self, code: str, message: str, details: list | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or []

    def to_dict(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class EquiReachValidationError(EquiReachError):
    def __init__(self, message: str, details: list | None = None):
        super().__init__("INVALID_INPUT", message, details)
