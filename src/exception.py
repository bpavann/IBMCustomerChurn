import sys
import traceback

def error_msg_details(error: Exception):
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_tb is not None:
        filename = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        filename = "Unknown"
        line_number = "Unknown"
    error_message = (
        f"\n[ERROR DETAILS]\n"
        f"File       : {filename}\n"
        f"Line       : {line_number}\n"
        f"Type       : {type(error).__name__}\n"
        f"Message    : {str(error)}\n"
        f"Traceback  :\n{traceback.format_exc()}"
    )
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message=error_msg_details(error_message,error_details=error_details)

    def __str__(self):
        return self.error_message
    
class CustomException(Exception):
    def __init__(self, error: Exception):
        super().__init__(str(error))
        self.error_message = error_msg_details(error)

    def __str__(self):
        return self.error_message