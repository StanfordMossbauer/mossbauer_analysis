

# This script is used to find the slow control files 

import mysql.connector
class sql_writer:
	def __init__(self,
				 host='192.168.2.2',
				 user='writer',
				 password='mossbauer_writer',
				 database='slowcontrol',
				 table='sc'):
		self.table = table  
		self.conn = mysql.connector.connect(
			host=host, user=user, password=password, database=database,
			autocommit=True, connection_timeout=5
		)
		self.cur = self.conn.cursor()
sql=sql_writer()


from datetime import UTC


def get_values_in_timerange(sql, t_start, t_end, value_col="Vpp_set", table="science_run1"):
    """
    Read all rows of one variable in a given UTC datetime range.

    Parameters
    ----------
    sql : sql_writer
    t_start : datetime
        will be converted into UTC datetime
    t_end : datetime
        will be converted into UTC datetime
    value_col : str
        Column name, e.g. "Vpp_set"
    table : str
        Table name, e.g. "science_run1"

    Returns
    -------
    times : list[datetime]
        UTC datetimes from SQL
    values : list
        Values of the selected column
    """
    if t_end < t_start:
        raise ValueError("t_end must be >= t_start")

    fmt = "%Y-%m-%d %H:%M:%S"

    # force to UTC naive strings for SQL query
    t0 = t_start.astimezone(UTC).replace(tzinfo=None) if t_start.tzinfo else t_start
    t1 = t_end.astimezone(UTC).replace(tzinfo=None) if t_end.tzinfo else t_end

    start_str = t0.strftime(fmt)
    end_str = t1.strftime(fmt)

    cmd = (
        f"SELECT `TIME`, `{value_col}` FROM `{table}` "
        f"WHERE `TIME` BETWEEN '{start_str}' AND '{end_str}' "
        f"ORDER BY `TIME`"
    )

    sql.cur.execute(cmd)
    rows = sql.cur.fetchall()

    if not rows:
        return [], []

    times = [r[0].replace(tzinfo=UTC) if r[0].tzinfo is None else r[0].astimezone(UTC) for r in rows]
    values = [r[1] for r in rows]
    return times, values