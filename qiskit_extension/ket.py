"""Module to store and check the ket notation for the state vector"""

import configparser
import importlib.resources as pkg_resources
import re

from qiskit_extension import config


class _ConfigMeta(type):
    __z0: str = "0"
    __z1: str = "1"
    __x0: str = "+"
    __x1: str = "-"
    __y0: str = "i"
    __y1: str = "j"

    def __new__(cls, name, bases, namespace):
        config_parser = configparser.ConfigParser()
        with pkg_resources.path(config, "config.ini") as path:
            config_parser.read(path)
        for ket in config_parser["ket"]:
            if ket not in ["z0", "z1", "x0", "x1", "y0", "y1"]:
                raise ValueError("Unknown basis.")
            if len(config_parser["ket"][ket]) != 1:
                raise ValueError("The ket notation should be a single character.")
            namespace[f"__{ket}"] = config_parser["ket"][ket]

        return super().__new__(cls, name, bases, namespace)

    @property
    def z0(self) -> str:
        return self.__z0

    @property
    def z1(self) -> str:
        return self.__z1

    @property
    def x0(self) -> str:
        return self.__x0

    @property
    def x1(self) -> str:
        return self.__x1

    @property
    def y0(self) -> str:
        return self.__y0

    @property
    def y1(self) -> str:
        return self.__y1


class Ket(metaclass=_ConfigMeta):
    """Class to store and check the ket notation for the state vector"""

    @classmethod
    def check_valid(cls, label: str) -> bool:
        """
        Check if the input label is valid.
        """
        ket_regex = "^[" + cls.z0 + cls.z1 + cls.x0 + cls.x1 + cls.y0 + cls.y1 + "]+$"
        # escape the minus sign in the regex
        # e.g. if ket notation = |0>, |1>, |+>, |->, |i>, |j>, then ket_str = R"^[01+\-rj]+$"
        ket_regex = ket_regex.replace("-", R"\-")

        if re.match(ket_regex, label) is None:
            return False
        return True
