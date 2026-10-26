import unittest

from lob_sim.core.lob import LimitOrderBook
from lob_sim.core.models import Order, Side


def _order(
    order_id: str,
    side: Side,
    price: float,
    quantity: float,
) -> Order:
    return Order.create(
        order_id=order_id,
        agent_id=f"agent-{order_id}",
        side=side,
        price=price,
        qty=quantity,
        timestamp=0.0,
    )


class SnapshotMemoizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.book = LimitOrderBook()
        self.book.place_limit_order(_order("bid-1", Side.BUY, 99.0, 10.0), 0.0)
        self.book.place_limit_order(_order("ask-1", Side.SELL, 101.0, 12.0), 0.0)

    def test_cached_snapshot_is_not_mutable_by_caller(self):
        bids, asks = self.book.get_snapshot(5)
        bids.append((98.0, 999.0))
        asks.clear()

        self.assertEqual(
            self.book.get_snapshot(5),
            ([(99.0, 10.0)], [(101.0, 12.0)]),
        )

    def test_limit_order_invalidates_every_cached_depth(self):
        self.book.get_snapshot(1)
        self.book.get_snapshot(5)
        self.book.place_limit_order(_order("bid-2", Side.BUY, 100.0, 7.0), 1.0)

        self.assertEqual(self.book.get_snapshot(1)[0], [(100.0, 7.0)])
        self.assertEqual(
            self.book.get_snapshot(5)[0],
            [(100.0, 7.0), (99.0, 10.0)],
        )

    def test_market_order_fill_invalidates_cached_quantities(self):
        self.book.get_snapshot(5)
        self.book.place_market_order(_order("market", Side.BUY, 0.0, 5.0), 1.0)

        self.assertEqual(self.book.get_snapshot(5)[1], [(101.0, 7.0)])

    def test_cancel_invalidates_cache_only_when_book_changes(self):
        expected = self.book.get_snapshot(5)
        self.assertFalse(self.book.cancel_order("missing"))
        self.assertEqual(self.book.get_snapshot(5), expected)

        self.assertTrue(self.book.cancel_order("bid-1"))
        self.assertEqual(self.book.get_snapshot(5)[0], [])

    def test_trim_invalidates_snapshot_after_removing_levels(self):
        self.book.place_limit_order(_order("bid-2", Side.BUY, 98.0, 8.0), 1.0)
        self.book.get_snapshot(5)
        removed = self.book.trim_to_depth(1)

        self.assertEqual([order.order_id for order in removed], ["bid-2"])
        self.assertEqual(self.book.get_snapshot(5)[0], [(99.0, 10.0)])


if __name__ == "__main__":
    unittest.main()
