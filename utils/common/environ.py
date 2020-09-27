
import os
from dotenv import load_dotenv, find_dotenv, dotenv_values

import logging
import tensorflow as tf


load_dotenv()


# 用來取得非字串形式或其他套件的值
__package_attr_map = {
    'LOG_LEVEL': logging.__dict__,

    'WORKING': {'True': True, 'False':False}
}

def env(var_name, default=None, dynamic=False):

    if not dynamic:
        var = os.getenv(var_name, default)

    else:
        # 有時在實驗中途時需要變更設定，可藉dynamic參數動態地讀入經中途更新過的.env，但需謹慎使用、盡量避免
        # 動態讀入的env只在讀入當下作用，不影響原本已讀入的env
        var = dotenv_values(find_dotenv()).get(var_name, default)

    return __package_attr_map[var_name].get(var, default) if var_name in __package_attr_map else var
