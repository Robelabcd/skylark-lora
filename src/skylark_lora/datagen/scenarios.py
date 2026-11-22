"""One builder per category. Each returns a training case whose answer is computed by code.

A case has the same shape as a test case: today, messages, expected call.
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Any

from skylark_lora.datagen import dates, text
from skylark_lora.datagen.places import (
    AMBIGUOUS,
    TRAIN_PLACES,
    Place,
    is_metro,
    name,
    pick_pair,
)
from skylark_lora.datagen.world import (
    Booking,
    Offer,
    clock,
    make_bookings,
    make_offers,
    short_city,
    summarize_bookings,
    summarize_offers,
)

Case = dict[str, Any]


def random_today(rng: random.Random) -> dt.date:
    """A conversation date between November 2026 and June 2027."""
    return dt.date(2026, 11, 1) + dt.timedelta(days=rng.randint(0, 241))


def call(name_: str, **arguments: Any) -> dict[str, Any]:
    return {"name": name_, "arguments": arguments}


def case(
    category: str,
    today: dt.date,
    messages: list[dict],
    expected: dict,
    notes: str | None = None,
) -> Case:
    out = {
        "category": category,
        "today": today.isoformat(),
        "messages": messages,
        "expected": expected,
    }
    if notes:
        out["notes"] = notes
    return out


def user(content: str) -> dict:
    return {"role": "user", "content": content}


# ---------- Search requests ----------

SEARCH_TEMPLATES = [
    "Find me flights from {o} to {d} {when}",
    "I need a flight from {o} to {d} {when}",
    "{o} to {d} {when}",
    "Can you search {o} to {d} {when}",
    "Any flights {o} to {d} {when}",
    "I'm flying {o} to {d} {when}. What's available?",
    "Show me flights from {o} to {d} {when}",
    "What's flying from {o} to {d} {when}",
    "Looking to get from {o} to {d} {when}",
    "I have to be in {d} {when}, leaving from {o}. Any options?",
    "Search {o} to {d} {when} please",
    "Could you check flights {o} to {d} {when}",
    "Heading to {d} from {o} {when}. Can you look?",
    "I want to fly out of {o} to {d} {when}",
    "Book me something from {o} to {d} {when}",
    "What are my options {o} to {d} {when}",
    "Need a ticket {o} to {d} {when}",
    "Trip from {o} to {d} {when}. What do you have?",
    "Going to {d} {when} from {o}. What flights are there?",
    "Please look up {o} to {d} {when}",
]
NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _adults_phrase(rng: random.Random, n: int) -> str:
    if n == 1:
        return rng.choice(
            ["just me", "for one adult", "1 passenger", "only one traveler"]
        )
    if n == 2 and rng.random() < 0.4:
        return rng.choice(
            [
                "me and my wife",
                "me and my husband",
                "me and my partner",
                "me and a friend",
            ]
        )
    word = rng.choice([str(n), NUMBER_WORDS[n]])
    return rng.choice(
        [
            f"for {word} adults",
            f"for {word} people",
            f"there are {word} of us",
            f"{word} passengers",
            f"{word} seats",
        ]
    )


def search_request(
    rng: random.Random,
    today: dt.date,
    *,
    relative: bool = False,
    options: int = 0,
    prefer_names: bool = False,
) -> tuple[str, dict[str, Any], list[str]]:
    """A search request, its correct arguments, and the place words used (for typos)."""
    o, d = pick_pair(rng)
    if prefer_names:
        while not (is_metro(o) or is_metro(d) or o.metro or d.metro):
            o, d = pick_pair(rng)
    o_name, d_name = (
        name(rng, o, allow_code=not prefer_names),
        name(rng, d, allow_code=not prefer_names),
    )
    phrase, date = dates.departure(rng, today, relative)
    args: dict[str, Any] = {
        "origin": o.code,
        "destination": d.code,
        "departure_date": date.isoformat(),
    }

    extras: list[str] = []
    kinds = rng.sample(["return", "adults", "cabin", "stops"], k=options)
    for kind in kinds:
        if kind == "return":
            r_phrase, r_date = dates.return_date(rng, date)
            args["return_date"] = r_date.isoformat()
            extras.append(
                rng.choice(["returning ", "back ", "coming back ", "return "])
                + text.when(r_phrase)
            )
        elif kind == "adults":
            n = rng.choice([1, 2, 2, 3, 4, 5, 6])
            if n > 1:
                args["adults"] = n
            extras.append(_adults_phrase(rng, n))
        elif kind == "cabin":
            cabin = rng.choice(["business", "premium", "economy"])
            if cabin != "economy":
                args["cabin"] = cabin
            extras.append(
                rng.choice(
                    {
                        "business": [
                            "business class",
                            "in business",
                            "business please",
                        ],
                        "premium": ["premium economy", "in premium economy"],
                        "economy": ["economy", "economy is fine", "coach is fine"],
                    }[cabin]
                )
            )
        else:
            stops = rng.choice([0, 0, 1, 2])
            args["max_stops"] = stops
            extras.append(
                rng.choice(
                    {
                        0: [
                            "nonstop",
                            "nonstop only",
                            "direct flights only",
                            "no layovers",
                        ],
                        1: [
                            "at most one stop",
                            "no more than one stop",
                            "one connection max",
                        ],
                        2: ["up to two stops", "two stops max"],
                    }[stops]
                )
            )

    base = rng.choice(SEARCH_TEMPLATES).format(
        o=o_name, d=d_name, when=text.when(phrase)
    )
    base = text.add_details(base, extras)
    if "return_date" in args and rng.random() < 0.3:
        first = base if base.startswith(("I ", "I'm ")) else base[0].lower() + base[1:]
        base = "Round trip: " + first
    return text.sentence(base), args, [o_name, d_name]


def search_case(rng: random.Random, category: str) -> Case:
    today = random_today(rng)
    if category == "simple_search":
        msg, args, _ = search_request(rng, today)
    elif category == "search_options":
        msg, args, _ = search_request(rng, today, options=rng.choice([1, 2, 2, 3]))
    elif category == "relative_date":
        msg, args, _ = search_request(
            rng, today, relative=True, options=rng.choice([0, 0, 1])
        )
    elif category == "city_airport":
        msg, args, _ = search_request(
            rng, today, prefer_names=True, options=rng.choice([0, 1])
        )
    else:  # messy
        raw, args, words = search_request(
            rng, today, relative=rng.random() < 0.3, options=rng.choice([0, 1, 2])
        )
        msg = text.messy(rng, raw, words)
        return case(category, today, [user(msg)], call("search_flights", **args))
    if rng.random() < 0.2:
        msg = text.casual(rng, msg)
    return case(category, today, [user(msg)], call("search_flights", **args))


# ---------- Conversation contexts ----------


def search_context(
    rng: random.Random, today: dt.date, n_offers: int | None = None
) -> tuple[list[dict], list[Offer]]:
    """An earlier turn: the user searched, the tool returned offers, the assistant summarized them."""
    o, d = pick_pair(rng)
    depart = today + dt.timedelta(days=rng.randint(3, 90))
    args: dict[str, Any] = {
        "origin": o.code,
        "destination": d.code,
        "departure_date": depart.isoformat(),
    }
    round_trip = rng.random() < 0.35
    request = f"Flights from {name(rng, o)} to {name(rng, d)} on {dates.absolute_phrase(rng, depart)}"
    if round_trip:
        back = depart + dt.timedelta(days=rng.randint(2, 12))
        args["return_date"] = back.isoformat()
        request += f", back {dates.absolute_phrase(rng, back)}"
    offers = make_offers(rng, depart, (o.code, d.code), n_offers)
    messages = [
        user(request),
        {"role": "assistant", "tool_call": call("search_flights", **args)},
        {
            "role": "tool",
            "name": "search_flights",
            "content": {"offers": [x.as_tool_result() for x in offers]},
        },
        {"role": "assistant", "content": summarize_offers(rng, offers, round_trip)},
    ]
    return messages, offers


BOOKINGS_REQUESTS = [
    "What trips do I have?",
    "Can you show my reservations?",
    "List my upcoming flights",
    "Pull up my bookings please",
    "What have I got booked?",
    "Show my trips",
]


def bookings_context(
    rng: random.Random, today: dt.date, n: int
) -> tuple[list[dict], list[Booking]]:
    bookings = make_bookings(rng, today, n)
    messages = [
        user(rng.choice(BOOKINGS_REQUESTS)),
        {"role": "assistant", "tool_call": call("list_bookings")},
        {
            "role": "tool",
            "name": "list_bookings",
            "content": {"bookings": [b.as_tool_result() for b in bookings]},
        },
        {"role": "assistant", "content": summarize_bookings(bookings)},
    ]
    return messages, bookings


def offer_references(offers: list[Offer]) -> list[tuple[str, Offer]]:
    """Every phrase that points to exactly one offer."""
    refs: list[tuple[str, Offer]] = []
    for o in offers:
        refs += [
            (f"the {o.airline} flight", o),
            (f"the {o.airline} one", o),
            (f"the {clock(o.depart)} one", o),
        ]
    by_price = sorted(offers, key=lambda o: o.price_usd)
    refs += [
        ("the cheapest one", by_price[0]),
        ("the lowest-priced option", by_price[0]),
    ]
    if len(offers) == 2:
        refs += [
            ("the cheaper one", by_price[0]),
            ("the more expensive one", by_price[-1]),
        ]
    nonstop = [o for o in offers if o.stops == 0]
    if len(nonstop) == 1:
        refs += [("the nonstop", nonstop[0]), ("the direct one", nonstop[0])]
    refs += [("the earliest one", offers[0]), ("the last one", offers[-1])]
    ordinals = ["first", "second", "third", "fourth"]
    refs += [(f"the {ordinals[i]} option", o) for i, o in enumerate(offers)]
    return refs


FARES = {
    "basic": ["basic", "basic economy", "the basic fare", "the cheapest fare type"],
    "main": ["main", "main cabin", "the regular fare", "standard"],
    "flex": ["flex", "the flexible fare", "the flex fare", "a flexible ticket"],
}


# ---------- Follow-ups on search results ----------


def follow_up_case(rng: random.Random) -> Case:
    today = random_today(rng)
    context, offers = search_context(rng, today)
    ref, offer = rng.choice(offer_references(offers))
    action = rng.choice(["details", "advice", "watch", "watch_target", "book", "chain"])

    if action == "details":
        msg = rng.choice(
            [
                "Tell me more about {r}",
                "What's included with {r}?",
                "Does {r} include a checked bag?",
                "Can I see the fare rules for {r}?",
                "What's the baggage policy on {r}?",
                "Give me the details on {r}",
            ]
        ).format(r=ref)
        return case(
            "follow_up",
            today,
            context + [user(text.sentence(msg))],
            call("get_offer_details", offer_id=offer.offer_id),
        )
    if action == "advice":
        msg = rng.choice(
            [
                "Should I book {r} now or wait?",
                "Is {r} a good price?",
                "Will {r} get cheaper?",
                "Buy now or wait on {r}?",
                "Do you think the price of {r} will drop?",
            ]
        ).format(r=ref)
        return case(
            "follow_up",
            today,
            context + [user(text.sentence(msg))],
            call("get_price_advice", offer_id=offer.offer_id),
        )
    if action == "watch":
        msg = rng.choice(
            [
                "Track the price of {r}",
                "Watch {r} for me",
                "Keep an eye on {r}",
                "Set a price alert on {r}",
            ]
        )
        return case(
            "follow_up",
            today,
            context + [user(text.sentence(msg.format(r=ref)))],
            call("create_price_watch", offer_id=offer.offer_id),
            "No target price given, so the target is left out.",
        )
    if action == "watch_target":
        target = max(50, round(offer.price_usd * rng.uniform(0.7, 0.95) / 10) * 10)
        msg = rng.choice(
            [
                "Alert me if {r} drops below ${t}",
                "Let me know when {r} goes under {t} dollars",
                "Watch {r} and tell me if it hits ${t}",
            ]
        ).format(r=ref, t=target)
        return case(
            "follow_up",
            today,
            context + [user(text.sentence(msg))],
            call("create_price_watch", offer_id=offer.offer_id, target_price=target),
        )
    if action == "book":
        fare = rng.choice(list(FARES))
        msg = rng.choice(
            [
                "Book {r} with {f}",
                "Let's go with {r}, {f} please",
                "I'll take {r}. {f} fare.",
                "Book {r}, {f}",
            ]
        ).format(r=ref, f=rng.choice(FARES[fare]))
        return case(
            "follow_up",
            today,
            context + [user(text.sentence(msg))],
            call("prepare_booking", offer_id=offer.offer_id, fare=fare),
        )

    # chain: price advice first, then act on "it"
    verdict = rng.choice(["wait", "buy"])
    advice_text = (
        "Prices on this route usually drop closer to the date, so waiting may save money."
        if verdict == "wait"
        else "This is a good price for this route; prices usually rise from here."
    )
    chain = [
        user(text.sentence(f"How's the price on {ref}?")),
        {
            "role": "assistant",
            "tool_call": call("get_price_advice", offer_id=offer.offer_id),
        },
        {
            "role": "tool",
            "name": "get_price_advice",
            "content": {"offer_id": offer.offer_id, "verdict": verdict},
        },
        {"role": "assistant", "content": advice_text},
    ]
    if verdict == "wait":
        last = rng.choice(
            [
                "Ok, watch it for me then",
                "Then set an alert on that one",
                "Fine, track it instead",
            ]
        )
        expected = call("create_price_watch", offer_id=offer.offer_id)
    else:
        fare = rng.choice(list(FARES))
        last = rng.choice(
            ["Great, book it with {f}", "Ok let's book that one, {f}"]
        ).format(f=rng.choice(FARES[fare]))
        expected = call("prepare_booking", offer_id=offer.offer_id, fare=fare)
    return case(
        "follow_up",
        today,
        context + chain + [user(text.sentence(last))],
        expected,
        "'It' is the offer from the previous turn.",
    )


# ---------- Bookings ----------

LIST_REQUESTS = [
    "What trips do I have coming up?",
    "Do I have any flights booked?",
    "Pull up my reservations",
    "What have I booked so far?",
    "Can you list my trips?",
    "I forgot what I booked. Can you check?",
    "Show all my upcoming flights",
]


def bookings_case(rng: random.Random) -> Case:
    today = random_today(rng)
    kind = rng.choice(["list", "cancel_unknown", "cancel_known", "cancel_known"])
    if kind == "list":
        return case(
            "bookings", today, [user(rng.choice(LIST_REQUESTS))], call("list_bookings")
        )
    if kind == "cancel_unknown":
        city = short_city(rng.choice([p for p in TRAIN_PLACES if not is_metro(p)]))
        msg = rng.choice(
            [
                "Cancel my {c} trip",
                "I need to cancel my flight to {c}",
                "Please cancel the {c} booking",
                "My {c} plans fell through, cancel it",
            ]
        ).format(c=city)
        return case(
            "bookings",
            today,
            [user(text.sentence(msg))],
            call("list_bookings"),
            "Booking ID unknown, so look it up first.",
        )

    context, bookings = bookings_context(rng, today, rng.choice([1, 2, 2, 3]))
    target = rng.choice(bookings)
    ref_kind = rng.choice(["city", "date", "airline"])
    if ref_kind == "city":
        ref = f"the {short_city(target.destination)} one"
    elif ref_kind == "date":
        ref = f"the {target.depart.strftime('%B %-d')} flight"
    else:
        ref = f"the {target.airline} booking"
    if len(bookings) == 1:
        ref = rng.choice([ref, "it", "that booking"])
    msg = rng.choice(
        ["Cancel {r}", "Please cancel {r}", "I can't make it anymore. Cancel {r}"]
    ).format(r=ref)
    return case(
        "bookings",
        today,
        context + [user(text.sentence(msg))],
        call("prepare_cancellation", booking_id=target.booking_id),
    )


# ---------- Clarify ----------


def clarify_case(rng: random.Random) -> Case:
    today = random_today(rng)
    kind = rng.choice(
        [
            "no_origin",
            "no_date",
            "vague_date",
            "no_destination",
            "no_return",
            "ambiguous_city",
            "no_fare",
            "which_offer",
            "which_nonstop",
            "which_booking",
        ]
    )
    o, d = pick_pair(rng)
    phrase, _ = dates.departure(rng, today, relative=rng.random() < 0.5)
    on = text.when(phrase)

    def ask(question: str) -> dict:
        return call("clarify", question=question)

    if kind == "no_origin":
        msg = rng.choice(
            [
                "Find me flights to {d} {w}",
                "I need to get to {d} {w}",
                "Any flights into {d} {w}",
                "Can you get me to {d} {w}",
            ]
        )
        return case(
            "clarify",
            today,
            [user(text.sentence(msg.format(d=name(rng, d), w=on)))],
            ask(
                rng.choice(
                    [
                        "Where will you be flying from?",
                        "Which city or airport are you leaving from?",
                    ]
                )
            ),
            "No origin.",
        )
    if kind == "no_date":
        msg = rng.choice(
            [
                "{o} to {d}",
                "Flights from {o} to {d}?",
                "I want to fly {o} to {d}",
                "Thinking about a trip from {o} to {d}. What's out there?",
            ]
        )
        return case(
            "clarify",
            today,
            [user(text.sentence(msg.format(o=name(rng, o), d=name(rng, d))))],
            ask(
                rng.choice(
                    [
                        "What date would you like to fly?",
                        "Which day do you want to leave?",
                    ]
                )
            ),
            "No date.",
        )
    if kind == "vague_date":
        month = dates.MONTHS[(today.month + rng.randint(0, 3)) % 12]
        vague = rng.choice(
            [
                f"sometime in {month}",
                "next month",
                "in a few weeks",
                "later this year",
                "around the holidays",
                f"in early {month}",
            ]
        )
        msg = rng.choice(
            [
                "{o} to {d} {v}",
                "I'd like to fly from {o} to {d} {v}",
                "Looking at {o} to {d}, {v}",
            ]
        )
        return case(
            "clarify",
            today,
            [user(text.sentence(msg.format(o=name(rng, o), d=name(rng, d), v=vague)))],
            ask("Which exact date would you like to fly?"),
            "A vague time is not a date.",
        )
    if kind == "no_destination":
        msg = rng.choice(
            [
                "What's the earliest flight out of {o} {w}",
                "I need to leave {o} {w}",
                "Get me out of {o} {w}",
            ]
        )
        return case(
            "clarify",
            today,
            [user(text.sentence(msg.format(o=name(rng, o), w=on)))],
            ask("Where would you like to fly to?"),
            "No destination.",
        )
    if kind == "no_return":
        msg = rng.choice(
            [
                "Round trip {o} to {d}, leaving {w}",
                "{o} to {d} and back, leaving {w}",
                "I need a round trip from {o} to {d}, going out {w}",
            ]
        )
        return case(
            "clarify",
            today,
            [user(text.sentence(msg.format(o=name(rng, o), d=name(rng, d), w=phrase)))],
            ask(
                rng.choice(
                    [
                        "When would you like to fly back?",
                        "What date would you like to return?",
                    ]
                )
            ),
            "Round trip without a return date.",
        )
    if kind == "ambiguous_city":
        amb = rng.choice(AMBIGUOUS)
        other = name(rng, o)
        msg = rng.choice(["{a} to {b} {w}", "Flights from {a} to {b} {w}"])
        pair = (amb["name"], other) if rng.random() < 0.5 else (other, amb["name"])
        first, second = amb["options"]
        return case(
            "clarify",
            today,
            [user(text.sentence(msg.format(a=pair[0], b=pair[1], w=on)))],
            ask(f"Do you mean {first} or {second}?"),
            f"{amb['name']} is ambiguous.",
        )
    if kind == "which_booking":
        context, bookings = bookings_context(rng, today, rng.choice([2, 3]))
        cities = [short_city(b.destination) for b in bookings]
        msg = rng.choice(
            [
                "Cancel my flight",
                "Cancel my booking",
                "I want to cancel my trip",
                "Please cancel it",
            ]
        )
        return case(
            "clarify",
            today,
            context + [user(msg)],
            ask(f"Which booking would you like to cancel: {text.join_or(cities)}?"),
            "Several bookings; which one is unclear.",
        )

    context, offers = search_context(rng, today, n_offers=rng.choice([2, 3, 4]))
    airlines = [o.airline for o in offers]
    if kind == "no_fare":
        ref, _ = rng.choice(offer_references(offers))
        msg = rng.choice(["Book {r}", "I'll take {r}", "Let's go with {r}"]).format(
            r=ref
        )
        return case(
            "clarify",
            today,
            context + [user(text.sentence(msg))],
            ask("Which fare would you like: basic, main, or flex?"),
            "Booking needs a fare.",
        )
    if kind == "which_nonstop":
        for o in offers[:2]:
            o.stops, o.via = 0, None
        context, _ = _replace_offers(rng, context, offers)
        nonstops = [o.airline for o in offers if o.stops == 0]
        msg = rng.choice(
            [
                "Book the nonstop",
                "I'll take the direct one",
                "Let's do the nonstop flight",
            ]
        )
        return case(
            "clarify",
            today,
            context + [user(msg)],
            ask(f"{text.join_and(nonstops)} are nonstop. Which one would you like?"),
            "More than one nonstop.",
        )
    msg = rng.choice(
        ["Book that one", "I'll take it", "Let's book it", "That one works, book it"]
    )
    return case(
        "clarify",
        today,
        context + [user(msg)],
        ask(f"Which flight would you like: {text.join_or(airlines)}?"),
        "Several offers; which one is unclear.",
    )


def _replace_offers(
    rng: random.Random, context: list[dict], offers: list[Offer]
) -> tuple[list[dict], list[Offer]]:
    """Rewrite the tool result and summary after offers were changed."""
    round_trip = "return_date" in context[1]["tool_call"]["arguments"]
    context = list(context)
    context[2] = {
        "role": "tool",
        "name": "search_flights",
        "content": {"offers": [x.as_tool_result() for x in offers]},
    }
    context[3] = {
        "role": "assistant",
        "content": summarize_offers(rng, offers, round_trip),
    }
    return context, offers


# ---------- Decline ----------

DECLINE_TOPICS = {
    "hotels": (
        [
            "Can you book me a hotel in {c}?",
            "Find me a hotel near the {c} airport",
            "I also need a place to stay in {c}",
        ],
        "Skylark only helps with flights, not hotels.",
    ),
    "cars": (
        [
            "Can you reserve a rental car in {c}?",
            "I'll need a car when I land in {c}. Can you book one?",
        ],
        "Skylark only helps with flights, not car rentals.",
    ),
    "rides": (
        [
            "Can you get me an Uber to the airport?",
            "Book a taxi to the {c} airport",
            "Order me a ride to the airport",
        ],
        "Skylark can't book ground transportation.",
    ),
    "documents": (
        [
            "Do I need a visa for {k}?",
            "What documents do I need to enter {k}?",
            "How do I renew my passport?",
        ],
        "Skylark can't advise on visas or travel documents; please check official government sources.",
    ),
    "weather": (
        ["What's the weather like in {c} this week?", "Will it rain in {c} tomorrow?"],
        "Skylark only helps with flights, not weather forecasts.",
    ),
    "food": (
        ["Any good restaurants near the {c} airport?", "Where should I eat in {c}?"],
        "Skylark only helps with flights, not restaurants.",
    ),
    "trains": (
        ["Can you book a train to {c}?", "Find me a bus to {c}"],
        "Skylark only books flights, not trains or buses.",
    ),
    "seats": (
        ["Can you move me to a window seat?", "Upgrade my seat to an aisle"],
        "Skylark can't change seat assignments yet.",
    ),
    "payment": (
        [
            "Can I give you my credit card number to pay?",
            "Take my card details and pay for it",
            "Use my debit card to pay right now",
        ],
        "Skylark never takes payment details in chat; payment happens securely in the app.",
    ),
    "off_topic": (
        [
            "Write me a poem about clouds",
            "Help me with my math homework",
            "Tell me a joke",
            "What's the capital of Australia?",
            "Translate 'good morning' into French",
        ],
        "Skylark is a flight assistant and can only help with flights.",
    ),
}
COUNTRIES = [
    "Japan",
    "Brazil",
    "Kenya",
    "India",
    "Germany",
    "Canada",
    "Mexico",
    "Egypt",
    "Ghana",
    "Spain",
]


def decline_case(rng: random.Random) -> Case:
    today = random_today(rng)
    phrasings, reason = DECLINE_TOPICS[rng.choice(list(DECLINE_TOPICS))]
    city = short_city(rng.choice([p for p in TRAIN_PLACES if not is_metro(p)]))
    msg = text.sentence(rng.choice(phrasings).format(c=city, k=rng.choice(COUNTRIES)))
    context = search_context(rng, today)[0] if rng.random() < 0.3 else []
    if rng.random() < 0.15:
        msg = text.casual(rng, msg)
    return case("decline", today, context + [user(msg)], call("decline", reason=reason))


# ---------- Respond ----------


def respond_case(rng: random.Random) -> Case:
    today = random_today(rng)
    if rng.random() < 0.25:
        context, bookings = bookings_context(rng, today, rng.choice([1, 2, 3]))
        b = rng.choice(bookings)
        city = short_city(b.destination)
        kind = rng.choice(["date", "airline", "count"])
        if kind == "date":
            msg, answer = (
                f"When is my {city} trip?",
                f"Your {city} trip departs on {b.depart.strftime('%B %-d')}.",
            )
        elif kind == "airline":
            msg, answer = (
                f"Which airline is my {city} flight on?",
                f"Your {city} flight is on {b.airline}.",
            )
        else:
            n = len(bookings)
            msg, answer = (
                "How many trips do I have?",
                f"You have {n} booking{'s' if n > 1 else ''}.",
            )
        return case(
            "respond", today, context + [user(msg)], call("respond", message=answer)
        )

    context, offers = search_context(rng, today)
    o = rng.choice(offers)
    kind = rng.choice(["price", "cheapest", "nonstop", "arrive", "depart", "layover"])
    if kind == "price":
        msg = rng.choice(
            [
                f"How much is the {o.airline} one?",
                f"What was the {o.airline} price again?",
            ]
        )
        answer = f"The {o.airline} flight is ${o.price_usd:,}."
    elif kind == "cheapest":
        best = min(offers, key=lambda x: x.price_usd)
        msg = rng.choice(
            [
                "Which one is cheapest?",
                "What's the lowest price?",
                "Which is the cheapest option?",
            ]
        )
        answer = f"{best.airline} is the cheapest at ${best.price_usd:,}."
    elif kind == "nonstop":
        nonstop = [x.airline for x in offers if x.stops == 0]
        msg = rng.choice(
            [
                "Which ones are nonstop?",
                "Are any of them direct?",
                "Is there a nonstop?",
            ]
        )
        if not nonstop:
            answer = "None of them are nonstop."
        elif len(nonstop) == 1:
            answer = f"Only {nonstop[0]} is nonstop."
        else:
            answer = f"{text.join_and(nonstop)} are nonstop."
    elif kind == "arrive":
        msg = rng.choice(
            [
                f"When does the {o.airline} flight land?",
                f"What time does {o.airline} get in?",
            ]
        )
        answer = f"The {o.airline} flight arrives at {clock(o.arrive)}."
    elif kind == "depart":
        msg = rng.choice(
            [
                f"What time does the {o.airline} one leave?",
                f"When does {o.airline} depart?",
            ]
        )
        answer = f"The {o.airline} flight departs at {clock(o.depart)}."
    else:
        msg = rng.choice(
            [
                f"Does the {o.airline} flight have a layover?",
                f"How many stops on {o.airline}?",
            ]
        )
        if o.stops == 0:
            answer = f"No, the {o.airline} flight is nonstop."
        elif o.via:
            answer = f"It has 1 stop, in {o.via}."
        else:
            answer = f"It has {o.stops} stops."
    return case(
        "respond", today, context + [user(msg)], call("respond", message=answer)
    )


BUILDERS = {
    "simple_search": lambda rng: search_case(rng, "simple_search"),
    "search_options": lambda rng: search_case(rng, "search_options"),
    "relative_date": lambda rng: search_case(rng, "relative_date"),
    "city_airport": lambda rng: search_case(rng, "city_airport"),
    "messy": lambda rng: search_case(rng, "messy"),
    "follow_up": follow_up_case,
    "bookings": bookings_case,
    "clarify": clarify_case,
    "decline": decline_case,
    "respond": respond_case,
}

MIX = {
    "simple_search": 0.10,
    "search_options": 0.10,
    "relative_date": 0.10,
    "city_airport": 0.08,
    "messy": 0.05,
    "follow_up": 0.12,
    "bookings": 0.07,
    "clarify": 0.16,
    "decline": 0.12,
    "respond": 0.10,
}


__all__ = ["BUILDERS", "MIX", "Place"]
