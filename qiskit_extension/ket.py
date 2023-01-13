"""Module to store and check the ket notation for the state vector"""

import configparser
import importlib.resources as pkg_resources
import re

from qiskit_extension import config


class _ConfigMeta(type):
    _z0: str
    _z1: str
    _x0: str
    _x1: str
    _y0: str
    _y1: str

    def __new__(cls, name, bases, namespace):
        config_parser = configparser.ConfigParser()
        with pkg_resources.path(config, "config.ini") as path:
            config_parser.read(path)
        namespace["_z0"] = config_parser["ket"]["z0"]
        namespace["_z1"] = config_parser["ket"]["z1"]
        namespace["_x0"] = config_parser["ket"]["x0"]
        namespace["_x1"] = config_parser["ket"]["x1"]
        namespace["_y0"] = config_parser["ket"]["y0"]
        namespace["_y1"] = config_parser["ket"]["y1"]
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
