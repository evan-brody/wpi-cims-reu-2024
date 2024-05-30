# @file gen_part_info.py
# @author Evan Brody
# @brief Generates a normalized SQL database based on part_info.csv

import sqlite3
import pandas as pd

def main():
    whole_df = pd.read_csv("part_info.csv")
    conn = sqlite3.connect("part_info.db")

    def exec_SQL(conn, query):
        conn.execute(query)
        conn.commit()

    exec_SQL(conn, "PRAGMA foreign_keys = ON")

    def all_rows_select(conn, query):
        res = conn.execute(query)
        for row in res:
            print(row)

    fails = whole_df["Failure Mode"].drop_duplicates().reset_index(drop=True)
    fails = pd.Series(sorted(fails))

    comps = whole_df["Component"].drop_duplicates().reset_index(drop=True)
    comps = pd.Series(sorted(comps))

    exec_SQL(conn, "DROP TABLE IF EXISTS comp_fails")
    exec_SQL(conn, "DROP TABLE IF EXISTS fail_modes")
    exec_SQL(conn, "DROP TABLE IF EXISTS components")

    def comp_setup():
        query = """
        CREATE TABLE components (
            id INT PRIMARY KEY,
            name TEXT
            )
        """

        exec_SQL(conn, query)

        for i, e in enumerate(comps):
            query = f"INSERT INTO components (id, name) VALUES ({i}, '{e}')"
            exec_SQL(conn, query)

        all_rows_select(conn, "SELECT * FROM components")

    def fail_setup():
        query = """
        CREATE TABLE fail_modes (
            id INT PRIMARY KEY,
            desc TEXT
        )
        """

        exec_SQL(conn, query)

        for i, e in enumerate(fails):
            query = f"INSERT INTO fail_modes (id, desc) VALUES ({i}, '{e}')"
            exec_SQL(conn, query)

        all_rows_select(conn, "SELECT * FROM fail_modes")

    def comp_fails_setup():
        comp_fails_srs = whole_df[["Component", "Failure Mode"]].drop_duplicates().reset_index(drop=True)
        comp_fails_srs = comp_fails_srs.sort_values(by=["Component", "Failure Mode"])

        query = """
        CREATE TABLE comp_fails (
            comp_id INT NOT NULL,
            fail_id INT NOT NULL,
            frequency INT DEFAULT 1,
            severity INT DEFAULT 1,
            detection INT DEFAULT 1,
            lower_bound REAL DEFAULT 0,
            best_estimate REAL DEFAULT 0,
            upper_bound REAL DEFAULT 0,
            mission_time REAL DEFAULT 1,
            FOREIGN KEY(comp_id) REFERENCES components(id),
            FOREIGN KEY(fail_id) REFERENCES fail_modes(id),
            PRIMARY KEY (comp_id, fail_id)
        )
        """

        exec_SQL(conn, query)

        for _, row in whole_df.iterrows():
            comp_id = comps[comps == row["Component"]].index.astype(int)
            comp_id = sum(comp_id)
            fail_id = fails[fails == row["Failure Mode"]].index.astype(int)
            fail_id = sum(fail_id)
            query = f"""
            INSERT INTO comp_fails (comp_id, fail_id)
            VALUES ({comp_id}, {fail_id})"""
            exec_SQL(conn, query)

        all_rows_select(conn, "SELECT * FROM comp_fails")

        query = """
        CREATE TABLE local_comp_fails (
            comp_id INT NOT NULL,
            fail_id INT NOT NULL,
            frequency INT DEFAULT 1,
            severity INT DEFAULT 1,
            detection INT DEFAULT 1,
            lower_bound REAL DEFAULT 0,
            best_estimate REAL DEFAULT 0,
            upper_bound REAL DEFAULT 0,
            mission_time REAL DEFAULT 1,
            FOREIGN KEY(comp_id) REFERENCES components(id),
            FOREIGN KEY(fail_id) REFERENCES fail_modes(id),
            PRIMARY KEY (comp_id, fail_id)
        )
        """

        exec_SQL(conn, query)

        for _, row in whole_df.iterrows():
            comp_id = comps[comps == row["Component"]].index.astype(int)
            comp_id = sum(comp_id)
            fail_id = fails[fails == row["Failure Mode"]].index.astype(int)
            fail_id = sum(fail_id)
            query = f"""
            INSERT INTO local_comp_fails (comp_id, fail_id)
            VALUES ({comp_id}, {fail_id})"""
            exec_SQL(conn, query)

        all_rows_select(conn, "SELECT * FROM local_comp_fails")

    comp_setup()
    fail_setup()
    comp_fails_setup()

if __name__ == "__main__":
    main()