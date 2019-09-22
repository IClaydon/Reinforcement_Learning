import attr


@attr.s
class Hotel(object):
    available_rooms = attr.ib()
    capacities = attr.ib()
    merchant_id = attr.ib()
