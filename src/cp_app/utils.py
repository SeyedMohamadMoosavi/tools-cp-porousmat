# -*- coding: utf-8 -*-

"""The utilities for ``cp_app``."""


def iter_together(path_left: str, path_right: str):
    """Open the two files, iterate over them, and zip them together.

    :param path_left: A path to a CSV file
    :param path_right: A path to a CSV file
    """
    with open(path_left) as left_file, open(path_right) as right_file:
        for left_line, right_line in zip(left_file, right_file):
            left_idx, left_value = left_line.strip().split(',')
            right_idx, right_value = right_line.strip().split(',')
            yield left_idx, left_value, right_value



