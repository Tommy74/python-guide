"""Tests for unpacking examples in pyguide.py."""

import pytest
from pyguide import (
    unpack_coordinates,
    unpack_rgb,
    first_and_rest,
    head_middle_tail,
    sum_pairs,
    swap,
    unpack_nested,
    min_max,
)


# --- Basic sequence unpacking ---

def test_unpack_coordinates():
    assert unpack_coordinates((3, 7)) == (3, 7)


def test_unpack_rgb():
    assert unpack_rgb((255, 128, 0)) == (255, 128, 0)


# --- Extended unpacking with * ---

def test_first_and_rest():
    first, rest = first_and_rest([1, 2, 3, 4])
    assert first == 1
    assert rest == [2, 3, 4]


def test_first_and_rest_single_element():
    first, rest = first_and_rest([42])
    assert first == 42
    assert rest == []


def test_head_middle_tail():
    head, middle, tail = head_middle_tail([1, 2, 3, 4, 5])
    assert head == 1
    assert middle == [2, 3, 4]
    assert tail == 5


def test_head_middle_tail_minimal():
    head, middle, tail = head_middle_tail([10, 20])
    assert head == 10
    assert middle == []
    assert tail == 20


# --- Unpacking in for loops ---

def test_sum_pairs():
    assert sum_pairs([(1, 2), (3, 4), (5, 6)]) == [3, 7, 11]


def test_sum_pairs_empty():
    assert sum_pairs([]) == []


# --- Swapping variables ---

def test_swap():
    assert swap(1, 2) == (2, 1)


def test_swap_strings():
    assert swap("hello", "world") == ("world", "hello")


# --- Nested unpacking ---

def test_unpack_nested():
    assert unpack_nested(((1, 2), 3)) == (1, 2, 3)


# --- Unpacking function return values ---

def test_min_max():
    lo, hi = min_max([4, 1, 7, 2, 9, 3])
    assert lo == 1
    assert hi == 9


def test_min_max_single():
    lo, hi = min_max([5])
    assert lo == 5
    assert hi == 5
