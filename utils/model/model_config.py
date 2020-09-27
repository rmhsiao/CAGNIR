
import dataclasses



class ModelConfig(object):

    def astuple(self):

        return dataclasses.astuple(self)

    def asdict(self):

        return dataclasses.asdict(self)
