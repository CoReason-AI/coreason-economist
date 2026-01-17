# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist


class BudgetExhaustedError(Exception):
    """
    Raised when a request exceeds the allocated budget.
    """

    def __init__(self, message: str, limit_type: str, limit_value: float, estimated_value: float) -> None:
        super().__init__(message)
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.estimated_value = estimated_value
