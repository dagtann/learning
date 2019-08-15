from typing import Tuple, Sequence, List, Any, Callable, Dict, Iterator
from collections import defaultdict

Row = Dict[str, Any]  # Database row
WhereClause = Callable[[Row], bool]  # Predicate for a single row
HavingClause = Callable[[List[Row]], bool]  # Predicate over multiple rows

# CREATE TABLE AND INSERT

class Table:
    def __init__(self, columns: List[str], types: List[type]) -> None:
        assert len(columns) == len(types), "# of cols must == # of types"

        self.columns = columns  # Names of columns
        self.types = types  # Data types of columns
        self.rows: List[Row] = []  # No data yet

    def col2type(self, col: str) -> type:  # helper function 2 get col type
        idx = self.columns.index(col)  # find column index
        return self.types[idx]  # and return index type

    def insert(self, values: list) -> None:
        # check right # of values
        if len(values) != len(self.types):
            raise ValueError(f"You need to provide {len(self.types)} values.")

        # check right type and values
        for value, typ3 in zip(values, self.types):
            if not isinstance(value, typ3) and value is not None:
                raise TypeError(f"Expected type {typ3} but got {value}.")

        # add rows to dict as "row"
        self.rows.append(dict(zip(self.columns, values)))

    def __getitem__(self, idx: int) -> Row:
        return self.rows[idx]

    def __iter__(self) -> Iterator[Row]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self):
        """Pretty representation of the table: columns, then rows"""
        rows = "\n".join(str(row) for row in self.rows)
        return f"{self.columns}\n{rows}"


users = Table(['user_id', 'name', 'num_friends'], [int, str, int])
users.insert([0, "Hero", 0])
users.insert([1, "Dunn", 2])
users.insert([2, "Sue", 3])
users.insert([3, "Chi", 2])
users.insert([4, "Thor", 3])
users.insert([5, "Clive", 2])
users.insert([6, "Hicks", 3])
users.insert([7, "Devin", 2])
users.insert([8, "Kate", 2])
users.insert([9, "Klein", 3])
users.insert([10, "Jen", 1])
print(users)

# UPDATE TABLE
# SQL example:
# UPDATE <table>
# SET <column> = <value>  # What field(s) to update and to what value(s)
# WHERE <condition>  # What rows to update


def update(self,
           updates: Dict[str, Any],
           predicate: WhereClause = lambda row: True):
    # (1) Check update validity
    for column, new_value in updates.items():
        if column not in self.columns:
            raise ValueError(f"invalid column: {column}")
        typ3 = self.col2type(column)
        if not isinstance(new_value, typ3):
            raise TypeError(f"Expected type {typ3}, but got {new_value}")

    # (2) execute update
    for row in self.rows:
        if predicate(row):
            for column, new_value in updates.items():
                row[column] = new_value


Table.update = update  # Add method to class

assert users[1]["num_friends"] == 2
users.update({"num_friends": 3},  # field to update
             lambda row: row["user_id"] == 1)  # row to update
assert users[1]["num_friends"] == 3

# DELETE FROM TABLE
# SQL Statement:
# DELETE FROM <table> WHERE <condition>
# NOTE: Unconditional DELETE statements delete the entire table


def delete(self, predicate: WhereClause = lambda row: True):
    """Delete all predicated rows from Table. Deletes the entire table if no
       predicate is supplied."""
    self.rows = [row for row in self.rows if not predicate(row)]

Table.delete = delete

users.delete(lambda row: row["user_id"] == 1)
print(users)
users.insert([1, "Dunn", 3])
print(users)

# SELECT FROM TABLE
# SQL EXAMPLES
# SELECT * FROM users
# SELECT user_id FROM users
# SELECT * FROM users WHERE user_id = 1
# Calculated statements
# SELECT LENGTH(name) FROM users WHERE name = "Dunn"

# Select method accepts:
# <keep_columns>: argument specifies columns to keep
# <additional_columns>: provide a dictionary of new column names and functions


def select(self,
           keep_columns: List[str] = None,
           additional_columns: Dict[str, Callable] = None) -> Table:

    if keep_columns is None:
        keep_columns = self.columns  # return all if none specified

    if additional_columns is None:
        additional_columns = {}

    # New column names and types
    new_columns = keep_columns + list(additional_columns.keys())
    keep_types = [self.col2type(col) for col in keep_columns]

    # Retrieve return type from type annotation
    add_types = [calculation.__annotations__['return'] for calculation
                 in additional_columns.values()]

    # Create new table to return results
    new_table = Table(new_columns, keep_types + add_types)

    for row in self.rows:
        new_row = [row[column] for column in keep_columns]
        for column_name, calculation in additional_columns.items():
            new_row.append(calculation(row))
        new_table.insert(new_row)

    return new_table


Table.select = select


def name_lengths(row) -> int:
    return (len(row["name"]))


users.select(additional_columns={"name_lengths": name_lengths})


def where(self, predicate: WhereClause = lambda row: True) -> Table:
    """Return only the rows that satisfy the supplied predicate"""
    where_table = Table(self.columns, self.types)
    for row in self.rows:
        if predicate(row):
            values = [row[column] for column in self.columns]
            where_table.insert(values)
    return where_table

def limit(self, num_rows: int) -> 'Table':
    """Return only the first `num_rows` rows"""
    limit_table = Table(self.columns, self.types)
    for i, row in enumerate(self.rows):
        if i >= num_rows:
            break
        values = [row[column] for column in self.columns]
        limit_table.insert(values)
    return limit_table


Table.where = where
Table.limit = limit

all_users = users.select()
assert len(all_users) == 11

all_users.limit(2)
dunn_ids = (
    users
    .where(lambda row: row["name"] == "Dunn")
    .select(keep_columns=["user_id"])
)