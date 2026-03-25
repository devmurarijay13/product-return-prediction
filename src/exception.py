import sys
import os
from src.logger import logging

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        self.error_message = error_message
        _,_,exc_tb = error_detail.exc_info()
        
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occured in python script name [{0}]  line numner [{1}] error message [{2}]".format(
        self.file_name,self.lineno,str(self.error_message)
    )
    

# if __name__ == "__main__":
#     try:
#         logging.info('entered the try block')
#         a = 1/0
#         print('this is will not print',a)
#     except Exception as e:
#         raise CustomException(e,sys)


