"""A small, reproducible flight world: search results and bookings for follow-up conversations.

IDs use ranges the test set never uses (OF-5000+, BK-2000+), so no test answer
can be memorized from training data.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass

from skylark_lora.datagen.places import ALL_PLACES, TRAIN_PLACES, Place, is_metro

AIRLINES = [
    "United",
    "Delta",
    "American",
    "Alaska",
    "JetBlue",
    "Southwest",
    "Frontier",
    "Hawaiian",
    "Air Canada",
    "British Airways",
    "Lufthansa",
    "Air France",
    "Iberia",
    "Emirates",
    "Qatar Airways",
    "Ethiopian",
    "Turkish Airlines",
    "Aeromexico",
]
HUBS = [
    "ATL",
    "DEN",
    "DFW",
    "ORD",
    "CLT",
    "PHX",
    "SEA",
    "MSP",
    "DTW",
    "IAH",
    "FRA",
    "MUC",
    "DOH",
]
_METRO_NAME = {p.code: p.refs[0] for p in ALL_PLACES if is_metro(p)}


def short_city(place: Place) -> str:
    """How a person names the city of a trip: "New York", "Denver", "Portland"."""
    if place.metro:
        return _METRO_NAME[place.metro]
    return place.city.split(",")[0]


@dataclass
class Offer:
    offer_id: str
    airline: str
    depart: dt.datetime
    arrive: dt.datetime
    stops: int
    via: str | None
    price_usd: int

    def as_tool_result(self) -> dict:
        row = {
            "offer_id": self.offer_id,
            "airline": self.airline,
            "depart": self.depart.strftime("%Y-%m-%dT%H:%M"),
            "arrive": self.arrive.strftime("%Y-%m-%dT%H:%M"),
            "stops": self.stops,
        }
        if self.via:
            row["via"] = self.via
        row["price_usd"] = self.price_usd
        return row


def clock(t: dt.datetime) -> str:
    """7:10 AM style."""
    return t.strftime("%I:%M %p").lstrip("0")


def make_offers(
    rng: random.Random, date: dt.date, route: tuple[str, str], n: int | None = None
) -> list[Offer]:
    """2 to 4 offers with different airlines, prices and departure times."""
    hubs = [h for h in HUBS if h not in route]
    n = n or rng.choice([2, 3, 3, 3, 4])
    airlines = rng.sample(AIRLINES, n)
    ids = rng.sample(range(5000, 10000), n)
    prices = rng.sample(range(89, 1400), n)
    hours = sorted(rng.sample(range(6, 22), n))
    offers = []
    for airline, oid, price, hour in zip(airlines, ids, prices, hours, strict=True):
        depart = dt.datetime.combine(
            date, dt.time(hour, rng.choice([0, 10, 15, 30, 45, 55]))
        )
        stops = rng.choice([0, 0, 1, 1, 2])
        arrive = depart + dt.timedelta(minutes=rng.randint(70, 330) + 90 * stops)
        via = rng.choice(hubs) if stops == 1 else None
        offers.append(Offer(f"OF-{oid}", airline, depart, arrive, stops, via, price))
    return offers


def describe_offer(o: Offer) -> str:
    stops = (
        "nonstop" if o.stops == 0 else f"{o.stops} stop" + ("s" if o.stops > 1 else "")
    )
    return f"{o.airline} {stops} at {clock(o.depart)} for ${o.price_usd:,}"


def summarize_offers(rng: random.Random, offers: list[Offer], round_trip: bool) -> str:
    noun = "round trips" if round_trip else "flights"
    parts = [describe_offer(o) for o in offers]
    listing = (
        ", ".join(parts[:-1]) + f", and {parts[-1]}"
        if len(parts) > 2
        else " and ".join(parts)
    )
    opener = rng.choice(
        [f"I found {len(offers)} {noun}:", f"Here are {len(offers)} {noun}:"]
    )
    return f"{opener} {listing}."


@dataclass
class Booking:
    booking_id: str
    origin: Place
    destination: Place
    airline: str
    depart: dt.date
    ret: dt.date | None

    def as_tool_result(self) -> dict:
        row = {
            "booking_id": self.booking_id,
            "trip": "round trip" if self.ret else "one way",
            "route": f"{self.origin.code}-{self.destination.code}",
            "depart": self.depart.isoformat(),
        }
        if self.ret:
            row["return"] = self.ret.isoformat()
        row["airline"] = self.airline
        return row


def make_bookings(rng: random.Random, today: dt.date, n: int) -> list[Booking]:
    """n upcoming bookings, each to a different destination city and on a different airline."""
    airports = [p for p in TRAIN_PLACES if not is_metro(p)]
    while True:
        destinations = rng.sample(airports, n)
        if len({short_city(d) for d in destinations}) == n:
            break
    origins = [
        rng.choice([p for p in airports if p.code != d.code]) for d in destinations
    ]
    airlines = rng.sample(AIRLINES, n)
    ids = rng.sample(range(2000, 10000), n)
    bookings = []
    for o, d, airline, bid in zip(origins, destinations, airlines, ids, strict=True):
        depart = today + dt.timedelta(days=rng.randint(3, 90))
        ret = (
            depart + dt.timedelta(days=rng.randint(2, 10))
            if rng.random() < 0.5
            else None
        )
        bookings.append(Booking(f"BK-{bid}", o, d, airline, depart, ret))
    return bookings


def summarize_bookings(bookings: list[Booking]) -> str:
    def one(b: Booking) -> str:
        when = b.depart.strftime("%B %-d")
        if b.ret:
            back = b.ret.strftime("%B %-d")
            return f"a round trip {b.origin.city} to {b.destination.city} on {b.airline}, {when} to {back}"
        return f"a one-way {b.origin.city} to {b.destination.city} on {b.airline} on {when}"

    if len(bookings) == 1:
        return f"You have 1 booking: {one(bookings[0])}."
    return (
        f"You have {len(bookings)} bookings: "
        + "; ".join(one(b) for b in bookings)
        + "."
    )
