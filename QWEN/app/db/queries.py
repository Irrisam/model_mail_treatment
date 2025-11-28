import psycopg2
import psycopg2.extras

def find_user_by_email(conn, email: str):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, email, name
            FROM medelse.user
            WHERE email = %s
            LIMIT 1
            """,
            (email,)
        )
        result = cur.fetchone()

    conn.close()

    return result  # peut être None
