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

    return result


def find_user_by_client_code(conn, client_code: str):
    """
    Recherche un utilisateur en base via son code client.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, email, name, client_code
            FROM medelse.user
            WHERE client_code = %s
            """,
            (client_code,),
        )
        row = cur.fetchone()

    if not row:
        return None

    return {
        "id": row[0],
        "email": row[1],
        "name": row[2],
        "client_code": row[3],
    }
