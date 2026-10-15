class RuntimeErrorBase(Exception):
    """Public, sanitised runtime error. Never include credentials or provider response bodies."""

    def __init__(self, code: str, message: str, status_code: int = 400, retriable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retriable = retriable

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "retriable": self.retriable}
