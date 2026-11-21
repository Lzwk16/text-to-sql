import sqlite3
from types import SimpleNamespace

import pytest


class FakeLLM:
    """Returns scripted responses in order; records prompts it was given."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def ainvoke(self, messages):
        self.prompts.append(messages[-1].content)
        text = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return SimpleNamespace(content=text)


@pytest.fixture
def shop_db(tmp_path):
    path = tmp_path / "shop.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            amount REAL
        );
        INSERT INTO customers VALUES (1,'Ann','Paris'),(2,'Bob','Rome');
        INSERT INTO orders VALUES (1,1,10.5),(2,1,20),(3,2,5);
        """
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def priced_db(tmp_path):
    """Prices stored as text with mixed currencies, like noshow.db."""
    path = tmp_path / "priced.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE bookings (id INTEGER PRIMARY KEY, room TEXT, price TEXT);
        INSERT INTO bookings VALUES
          (1,'King','SGD$ 100.00'),(2,'Queen','USD$ 100.00'),(3,'Single',NULL);
        """
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def wide_db(tmp_path):
    """Database with more tables than the full-schema threshold."""
    path = tmp_path / "wide.db"
    conn = sqlite3.connect(path)
    for i in range(12):
        conn.execute(f"CREATE TABLE t{i} (id INTEGER PRIMARY KEY, v TEXT)")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def fk_db(tmp_path):
    """12 tables: a customers->orders->order_items chain plus unrelated filler."""
    path = tmp_path / "fk.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE products (id INTEGER PRIMARY KEY, title TEXT, unit_price REAL);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER,
            order_date TEXT, FOREIGN KEY (customer_id) REFERENCES customers(id));
        CREATE TABLE order_items (id INTEGER PRIMARY KEY, order_id INTEGER,
            product_id INTEGER, quantity INTEGER,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id));
        """
    )
    for i in range(8):
        conn.execute(f"CREATE TABLE filler{i} (id INTEGER PRIMARY KEY, note TEXT)")
    conn.commit()
    conn.close()
    return path
