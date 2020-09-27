
from .environ import env
from .helpers import os_shutdown
from .mlogging import mlogging

import traceback
import winsound


class ExceptiontContainer(object):

    def __init__(self, log_name='exception.log', log_prefix=None, use_console=True,
                 to_raise=False, beep=True, shutdown=False, hibernate=False):

        self._logger = mlogging.get_logger(log_name, prefix=log_prefix, use_console=use_console)

        self.to_raise = to_raise
        self.beep = beep

        self._shutdown_opts = {'shutdown':shutdown, 'hibernate':hibernate}

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):

        if exc_type is not None:

            self._logger.error('%s: %s\n%s'%(exc_type.__name__, str(exc_val), traceback.format_exc()))

            if self.beep:
                winsound.Beep(frequency=522, duration=2020)

            if env('WORKING', dynamic=True)==False:
                os_shutdown(**self._shutdown_opts)

            if not self.to_raise:
                return True
