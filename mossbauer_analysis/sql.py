import mysql.connector
from datetime import timezone

UTC = timezone.utc

class sql:
    def __init__(self,
                 host='192.168.2.2',
                 user='writer',
                 password='mossbauer_writer',
                 database='slowcontrol',
                 table='science_run1'):
        self.table = table
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            autocommit=True,
            connection_timeout=5
        )
        self.cur = self.conn.cursor()


sql_writer = sql




def get_values_in_timerange(db, t_start, t_end,
                            value_cols=("sp_current_set", "Vpp_set"),
                            table="science_run1"):
    if t_end < t_start:
        raise ValueError("t_end must be >= t_start")

    if isinstance(value_cols, str):
        value_cols = (value_cols,)

    fmt = "%Y-%m-%d %H:%M:%S"

    # SQL stores naive UTC timestamps
    t0 = t_start.astimezone(UTC).replace(tzinfo=None)
    t1 = t_end.astimezone(UTC).replace(tzinfo=None)

    start_str = t0.strftime(fmt)
    end_str = t1.strftime(fmt)

    col_sql = ", ".join(f"`{c}`" for c in value_cols)

    cmd = (
        f"SELECT `TIME`, {col_sql} FROM `{table}` "
        f"WHERE `TIME` >= '{start_str}' AND `TIME` < '{end_str}' "
        f"ORDER BY `TIME`"
    )

    db.cur.execute(cmd)
    rows = db.cur.fetchall()

    if not rows:
        return [], {c: [] for c in value_cols}

    times = [
        r[0].replace(tzinfo=UTC) if r[0].tzinfo is None else r[0].astimezone(UTC)
        for r in rows
    ]

    data = {}
    for j, col in enumerate(value_cols, start=1):
        data[col] = [r[j] for r in rows]

    return times, data