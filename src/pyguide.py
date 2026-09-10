"""Examples of unpacking in Python."""


# --- Basic sequence unpacking ---

def unpack_coordinates(point):
    """Unpack a (x, y) tuple into named variables."""
    x, y = point
    return x, y


def unpack_rgb(color):
    """Unpack a (r, g, b) tuple."""
    r, g, b = color
    return r, g, b


# --- Extended unpacking with * ---

def first_and_rest(items):
    """Return the first element and the remaining ones separately."""
    first, *rest = items
    return first, rest


def head_middle_tail(items):
    """Split a sequence into head, middle, and tail."""
    head, *middle, tail = items
    return head, middle, tail


# --- Unpacking in for loops ---

def sum_pairs(pairs):
    """Sum each (a, b) pair and return a list of results."""
    return [a + b for a, b in pairs]


# --- Swapping variables ---

def swap(a, b):
    """Swap two values using tuple unpacking."""
    a, b = b, a
    return a, b


# --- Nested unpacking ---

def unpack_nested(data):
    """Unpack a nested structure like ((x, y), z)."""
    (x, y), z = data
    return x, y, z


# --- Unpacking function return values ---

def min_max(numbers):
    """Return (minimum, maximum) of a sequence."""
    return min(numbers), max(numbers)
