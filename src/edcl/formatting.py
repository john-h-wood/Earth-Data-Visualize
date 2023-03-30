"""
The edcl_formatting module houses all functions for converting information between formats.
"""

from .types import *

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']


def convert_month(value: str | int) -> str | int:
    """
    Converts a month number to a month name and vice-versa.

    Args:
        value: Either the month number or month name.

    Examples:
        convert_month(4) -> April
        convert_month('April') -> 4

    Returns:
        Either the month number or the month name.
    """
    if isinstance(value, str):
        return MONTHS.index(value) + 1
    else:
        return MONTHS[value - 1]


def tuple_to_string(values: tuple) -> str:
    """
    Returns a formatted string representation of a tuple of elements.

    Formats the elements in English form, with the Oxford comma. If the tuple is empty, 'No value' is returned. If
    the tuple contains a single element, that element is returned as a string.

    Examples:
        ('A', 'B', 'C') -> 'A, B and C'
        (1) -> '1'
        () -> 'No value'

    Args:
        values: The values.

    Returns:
        The string representation.
    """

    if len(values) == 0:
        return 'No value'
    elif len(values) == 1:
        return str(values[0])
    else:
        word = str()
        for i in range(len(values) - 1):
            word += str(values[i]) + ', '
        word += 'and ' + str(values[-1])
        return word


def format_month(month: int) -> str:
    """
    Formats a month number as a two-character-long string.

    Particularly useful for file path generation where months are represented with two characters.

    Example:
        The input 3 returns '03' while 12 returns '12'.

    Args:
        month: The month.

    Returns:
        The formatted string.
    """

    res = str(month)
    return '0' + res if len(res) == 1 else res


def time_to_suffix(time: TIME) -> str:
    """
    Returns a title suffix for a period of time, or at specific time.

    Uses time period convention detailed in the documentation for _get_data.

    Examples:
        An example is given for each possible outcome:
        year: None,   month: 4,      day: None,   hour: None   -> 'over April's'
        year: None,   month: None,   day: None,   hour: None   -> 'over all time'
        year: 2006,   month: None,   day: None,   hour: None   -> 'over 2006'
        year: 2007,   month: 3,      day: None,   hour: None   -> 'over March 2007'
        year: 2012,   month: 2,      day: 20,     hour: None   -> 'over February  20, 2012'
        year: 2018,   month: 6,      day: 11,     hour: 12     -> 'on June 11, 2018 at 12H'

    Args:
        time: The time period as (year, month, day, hour), all elements possibly being None.

    Returns:
        The suffix.
    """
    year, month, day, hour = time

    if (year is None) and (month is not None):
        return f'over {convert_month(month)}\'s'
    elif year is None:
        return 'over all time'
    elif month is None:
        return f'over {year}'
    elif day is None:
        return f'over {convert_month(month)} {year}'
    elif hour is None:
        return f'over {convert_month(month)} {day}, {year}'
    elif all((year is not None, month is not None, day is not None, hour is not None)):
        return f'on {convert_month(month)} {day}, {year} at {hour}H'


def time_stamps_to_filenames(time_stamps: TIME_STAMPS) -> tuple[str]:
    """
    Converts time stamps to file names (without an extension).

    Useful for generating automatic file names for plotting.

    Args:
        time_stamps: The time stamps.

    Returns:
        The file names.
    """
    names = list()

    for stamp in time_stamps:
        name = str()
        for piece in stamp:
            if piece is None:
                name += '-_'
            else:
                name += f'{str(piece)}_'

        names.append(name[:-1])

    return tuple(names)


def format_coordinates(latitude: float, longitude: float) -> str:
    """
    Formats a point's coordinates to a string with degree symbol.

    Examples:
        -9.6, 80 -> 9.6° S, 80° E
        10.2, -5.1 -> 10.2° N, 5.1° W
    Args:
        latitude: The latitude.
        longitude: The longitude.

    Returns:
        The formatted coordinate.
    """
    result = str()

    if latitude >= 0:
        result += f'{latitude}\N{DEGREE SIGN} N, '
    else:
        result += f'{abs(latitude)}\N{DEGREE SIGN} S, '

    if longitude >= 0:
        result += f'{longitude}\N{DEGREE SIGN} E'
    else:
        result += f'{abs(longitude)}\N{DEGREE SIGN} W'

    return result
