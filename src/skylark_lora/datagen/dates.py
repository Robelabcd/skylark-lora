"""Date phrases and their correct answers, computed by code.

Every phrase comes with the exact date it means, using the same rules the
model is given: a date that has already passed this month or year means the
next one, and "this Friday" is the next Friday after today.
"""

from __future__ import annotations

import datetime as dt
import random

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def ordinal(n: int) -> str:
    suffix = (
        "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )
    return f"{n}{suffix}"


def next_month_day(after: dt.date, month: int, day: int) -> dt.date:
    """The first (month, day) on or after `after`: this year, else next year."""
    candidate = dt.date(after.year, month, day)
    return candidate if candidate >= after else dt.date(after.year + 1, month, day)


def next_day_of_month(after: dt.date, day: int) -> dt.date:
    """The first date on or after `after` whose day of the month is `day`."""
    year, month = after.year, after.month
    for _ in range(3):
        try:
            candidate = dt.date(year, month, day)
        except ValueError:
            candidate = None
        if candidate and candidate >= after:
            return candidate
        month += 1
        if month == 13:
            year, month = year + 1, 1
    raise ValueError(f"no day {day} after {after}")


def next_weekday(after: dt.date, weekday: int) -> dt.date:
    """The next given weekday strictly after `after` (1 to 7 days later)."""
    days = (weekday - after.weekday()) % 7 or 7
    return after + dt.timedelta(days=days)


def absolute_phrase(rng: random.Random, date: dt.date) -> str:
    """Write a specific date the way people type it (month and day, no year)."""
    month, day = MONTHS[date.month - 1], date.day
    return rng.choice(
        [
            f"{month} {day}",
            f"{month} {ordinal(day)}",
            f"{month[:3]} {day}",
            f"{month[:3]} {ordinal(day)}",
            f"{date.month}/{day}",
            f"the {ordinal(day)} of {month}",
        ]
    )


def departure(
    rng: random.Random, today: dt.date, relative: bool
) -> tuple[str, dt.date]:
    """A departure phrase and the date it means.

    relative=False: a month-and-day phrase. relative=True: tomorrow, a weekday,
    "the 20th", "in 10 days" and similar, which must be worked out from today.
    """
    if not relative:
        date = today + dt.timedelta(days=rng.randint(1, 150))
        phrase = absolute_phrase(rng, date)
        assert _resolve_absolute(today, date) == date
        return phrase, date

    kind = rng.choice(
        ["today", "tomorrow", "after_tomorrow", "weekday", "day_of_month", "in_days"]
    )
    if kind == "today":
        return rng.choice(["today", "later today", "this evening"]), today
    if kind == "tomorrow":
        return rng.choice(
            ["tomorrow", "tomorrow morning", "tmrw"]
        ), today + dt.timedelta(days=1)
    if kind == "after_tomorrow":
        return "the day after tomorrow", today + dt.timedelta(days=2)
    if kind == "weekday":
        # Today's own weekday is skipped: "this Monday" said on a Monday is ambiguous.
        weekday = rng.choice([d for d in range(7) if d != today.weekday()])
        name = WEEKDAYS[weekday]
        phrase = rng.choice([f"this {name}", f"on {name}", f"this coming {name}", name])
        return phrase, next_weekday(today, weekday)
    if kind == "day_of_month":
        day = rng.randint(1, 28)
        return rng.choice(
            [f"the {ordinal(day)}", f"on the {ordinal(day)}"]
        ), next_day_of_month(today, day)
    days = rng.choice([3, 5, 7, 10, 14])
    words = {3: "three", 5: "five", 7: "a week", 10: "ten", 14: "two weeks"}[days]
    if days in (7, 14):
        phrase = rng.choice([f"{words} from today", f"in {words}", f"{words} from now"])
    else:
        phrase = rng.choice(
            [f"in {days} days", f"{words} days from now", f"{days} days from today"]
        )
    return phrase, today + dt.timedelta(days=days)


def return_date(rng: random.Random, depart: dt.date) -> tuple[str, dt.date]:
    """A return phrase and the date it means, always after the departure."""
    kind = rng.choice(["absolute", "day_of_month", "weekday"])
    if kind == "weekday":
        weekday = rng.randrange(7)
        return WEEKDAYS[weekday], next_weekday(depart, weekday)
    date = depart + dt.timedelta(days=rng.randint(2, 16))
    if kind == "day_of_month":
        return f"the {ordinal(date.day)}", date
    return absolute_phrase(rng, date), date


def _resolve_absolute(today: dt.date, date: dt.date) -> dt.date:
    return next_month_day(today, date.month, date.day)
