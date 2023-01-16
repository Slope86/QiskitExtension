"""Module to store and check the ket notation for the state vector"""

import configparser
import importlib.resources as pkg_resources
import re

from qiskit_extension import config


class _ConfigMeta(type):
    _z0: str = "0"
    _z1: str = "1"
    _x0: str = "+"
    _x1: str = "-"
    _y0: str = "i"
    _y1: str = "j"

    def __new__(cls, name, bases, namespace):
        config_parser = configparser.ConfigParser()
        with pkg_resources.path(config, "config.ini") as path:
            config_parser.read(path)
        for ket in config_parser["ket"]:
            if ket not in ["z0", "z1", "x0", "x1", "y0", "y1"]:
                raise ValueError("Unknown basis.")
            if len(config_parser["ket"][ket]) != 1:
                raise ValueError("The ket notation should be a single character.")
            namespace[f"_{ket}"] = config_parser["ket"][ket]

        return super().__new__(cls, name, bases, namespace)

    @property
    def z0(self) -> str:
        return self._z0

    @property
    def z1(self) -> str:
        return self._z1

    @property
    def x0(self) -> str:
        return self._x0

    @property
    def x1(self) -> str:
        return self._x1

    @property
    def y0(self) -> str:
        return self._y0

    @property
    def y1(self) -> str:
        return self._y1


class Ket(metaclass=_ConfigMeta):
    """Class to store and check the ket notation for the state vector"""

    @classmethod
    def check_valid(cls, label: str) -> bool:
        """
        Check if the input label is valid.
        """
        ket_list = ["^["]
        for ket in list(cls.__dict__.values())[3:9]:
            ket_list.append(ket)
        ket_list.append("]+$")
        ket_str = "".join(ket_list)
        ket_str = ket_str.replace("-", R"\-")

        # e.g. if ket notation = |0>, |1>, |+>, |->, |i>, |j>, then the ket_str = R"^[01+\-rj]+$"
        if re.match(ket_str, label) is None:
            return False
        return True
