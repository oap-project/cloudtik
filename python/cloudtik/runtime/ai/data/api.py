from enum import Enum, auto


class DataAPIType(Enum):
    PANDAS = auto()
    SPARK = auto()
    MODIN = auto()

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_str(api_impl):
        if api_impl.lower() == "pandas":
            return DataAPIType.PANDAS
        elif api_impl.lower() == "spark":
            return DataAPIType.SPARK
        elif api_impl.lower() == "modin":
            return DataAPIType.MODIN
        else:
            options = [e.name for e in DataAPIType]
            raise ValueError("Unsupported Data API type: {} (Select from: {})".format(
                api_impl, options))


class DataAPI(object):
    """The Data API class wraps all the implementation details of a Pandas Data API implementation.
    So that the application layer of using the DataAPI will not need to handle implementation specific
    things so that we can switch the implementation easily.
    """

    def __init__(self, api_type: DataAPIType | str):
        if type(api_type) is str:
            self.api_type = DataAPIType.from_str(api_type)
        else:
            self.api_type = api_type

    def pandas(self):
        return self.pandas_api()()

    def pandas_api(self):
        if self.api_type == DataAPIType.SPARK:
            def fn():
                import pyspark.pandas as pd
                return pd
            return fn
        elif self.api_type == DataAPIType.MODIN:
            def fn():
                import modin.pandas as pd
                return pd
            return fn
        else:
            def fn():
                import pandas as pd
                return pd
            return fn

    @property
    def native(self):
        return True if self.api_type == DataAPIType.PANDAS else False


def get_data_api(api_type):
    """Return a Data API object based on the type"""
    return DataAPI(api_type)
