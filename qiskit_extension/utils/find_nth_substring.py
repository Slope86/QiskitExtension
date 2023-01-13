def find_nth_substring(string: str, sub_string, n) -> int:
    """Find the index of nth substring in a string.

    Example:
        >>> find_nth_substring('123abcabcabc', 'abc', 2)
        6

    Args:
        string (str): The string to search.
        sub_string (_type_): The substring to search for.
        n (_type_): The nth substring to find.

    Returns:
        int: The index of the nth substring.
    """
    sub_string_index = string.find(sub_string)
    while sub_string_index >= 0 and n > 1:
        sub_string_index = string.find(sub_string, sub_string_index + len(sub_string))
        n -= 1
    return sub_string_index


if __name__ == "__main__":
    print(find_nth_substring("123abcabcabc", "abc", 2))
