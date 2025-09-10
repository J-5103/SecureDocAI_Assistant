# src/utils/db_handler.py
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Iterable, Tuple

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from src.logger import logging
from src.exception import CustomException
from src.utils.db_utils import get_database_connection


class DBHandler:
    """
    Central DB helper:
      - Ensures schema on init
      - Save documents / processed text
      - Save chat sessions & messages (Chat + Viz)
      - Save business-card contacts (normalized) + sync to flat table
      - NEW: List/export contacts per chat to Excel
      - NEW: Basic getters for chat + messages
    """
    def __init__(self):
        try:
            self.conn = get_database_connection()  # psycopg2 connection
            self.conn.autocommit = False
            self._ensure_schema()
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------ internals ------------------

    def _cursor(self):
        try:
            return self.conn.cursor()
        except Exception:
            # reconnect once if connection dropped
            self.conn = get_database_connection()
            self.conn.autocommit = False
            return self.conn.cursor()

    def _ensure_schema(self):
        """
        Create/upgrade schema pieces used by both Chat & Viz apps.
        """
        ddl = """
        CREATE EXTENSION IF NOT EXISTS pgcrypto;

        /* ------------ generic document metadata ------------ */
        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY,
          file_name TEXT NOT NULL,
          file_path TEXT NOT NULL,
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS processed_documents (
          document_id TEXT PRIMARY KEY,
          cleaned_text TEXT,
          table_text TEXT,
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        /* ------------ chat sessions & messages (BOTH Chat & Viz) ------------ */
        CREATE TABLE IF NOT EXISTS chat_sessions (
          id TEXT PRIMARY KEY,
          source TEXT NOT NULL DEFAULT 'chat',      -- 'chat' | 'viz'
          name TEXT,
          created_at TIMESTAMPTZ,
          last_message TEXT,
          message_count INTEGER NOT NULL DEFAULT 0,
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
          id TEXT PRIMARY KEY,
          chat_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
          source TEXT NOT NULL DEFAULT 'chat',      -- 'chat' | 'viz'
          sender TEXT NOT NULL,                     -- 'user' | 'ai'
          text TEXT,
          image_url TEXT,
          image_urls TEXT[],
          image_alt TEXT,
          status TEXT,
          ts TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_chat_messages_chat ON chat_messages (chat_id, ts DESC);

        /* ------------ business card (normalized model) ------------ */
        CREATE TABLE IF NOT EXISTS card_contacts (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          chat_id TEXT,
          source TEXT NOT NULL DEFAULT 'viz',       -- 'chat' | 'viz'
          raw_text TEXT,
          attachment_urls TEXT[],
          created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS card_names (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          contact_id UUID NOT NULL REFERENCES card_contacts(id) ON DELETE CASCADE,
          name TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS card_titles (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          contact_id UUID NOT NULL REFERENCES card_contacts(id) ON DELETE CASCADE,
          title TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS card_companies (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          contact_id UUID NOT NULL REFERENCES card_contacts(id) ON DELETE CASCADE,
          company TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS card_phones (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          contact_id UUID NOT NULL REFERENCES card_contacts(id) ON DELETE CASCADE,
          phone TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS card_emails (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          contact_id UUID NOT NULL REFERENCES card_contacts(id) ON DELETE CASCADE,
          email TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS card_websites (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          contact_id UUID NOT NULL REFERENCES card_contacts(id) ON DELETE CASCADE,
          url TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS card_addresses (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          contact_id UUID NOT NULL REFERENCES card_contacts(id) ON DELETE CASCADE,
          street TEXT, city TEXT, state TEXT, postal_code TEXT, country TEXT,
          full_text TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_card_contacts_created ON card_contacts (created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_card_phones_phone ON card_phones (phone);
        CREATE INDEX IF NOT EXISTS idx_card_emails_email ON card_emails (email);

        /* ------------ optional flat (1 row per contact) ------------ */
        CREATE TABLE IF NOT EXISTS business_contacts_flat (
          contact_id UUID PRIMARY KEY REFERENCES card_contacts(id) ON DELETE CASCADE,
          chat_id TEXT,
          source TEXT,
          names TEXT[],
          titles TEXT[],
          companies TEXT[],
          phones TEXT[],
          emails TEXT[],
          websites TEXT[],
          addresses TEXT[],
          raw_text TEXT,
          attachment_urls TEXT[],
          created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
        cur = self._cursor()
        try:
            cur.execute(ddl)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise CustomException(e, sys)
        finally:
            cur.close()

    # ------------------ documents / processed text ------------------

    def insert_metadata(self, filename: str, filepath: str) -> str:
        """
        Upsert into documents.
        """
        try:
            base = filename.rsplit(".", 1)[0]
            document_id = base.replace(" ", "_")
            q = """
            INSERT INTO documents (id, file_name, file_path)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE
              SET file_name = EXCLUDED.file_name,
                  file_path = EXCLUDED.file_path
            """
            cur = self._cursor()
            cur.execute(q, (document_id, filename, filepath))
            self.conn.commit()
            cur.close()
            return document_id
        except Exception as e:
            self.conn.rollback()
            raise CustomException(e, sys)

    def get_file_path(self, document_id: str) -> str:
        try:
            q = "SELECT file_path FROM documents WHERE id = %s"
            cur = self._cursor()
            cur.execute(q, (document_id,))
            row = cur.fetchone()
            cur.close()
            if row:
                return row[0]
            raise Exception("Document not found")
        except Exception as e:
            raise CustomException(e, sys)

    def save_to_database(self, data: dict, document_id: str = "unknown"):
        """
        Upsert cleaned text/table for a doc.
        """
        try:
            text = data.get("text", "") or ""
            table = data.get("table", pd.DataFrame())
            table_text = ""
            if isinstance(table, pd.DataFrame) and not table.empty:
                table_text = table.to_string(index=False)

            q = """
                INSERT INTO processed_documents (document_id, cleaned_text, table_text, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (document_id) DO UPDATE
                SET cleaned_text = EXCLUDED.cleaned_text,
                    table_text  = EXCLUDED.table_text,
                    updated_at  = now()
            """
            cur = self._cursor()
            cur.execute(q, (document_id, text, table_text))
            self.conn.commit()
            cur.close()

            logging.info(f"💾 Saved cleaned data for doc={document_id} (len={len(text)})")
        except Exception as e:
            self.conn.rollback()
            raise CustomException(e, sys)

    # ------------------ chat sessions & messages (Chat + Viz) ------------------

    def upsert_chat_session(
        self,
        chat_id: str,
        source: str,                 # 'chat' or 'viz'
        name: Optional[str],
        created_at_iso: Optional[str],
        last_message: Optional[str],
        message_count: Optional[int]
    ):
        """
        Save or update a chat session row.
        """
        try:
            q = """
            INSERT INTO chat_sessions (id, source, name, created_at, last_message, message_count, updated_at)
            VALUES (%s, %s, %s, %s, %s, COALESCE(%s,0), now())
            ON CONFLICT (id) DO UPDATE
              SET source        = EXCLUDED.source,
                  name          = EXCLUDED.name,
                  created_at    = COALESCE(chat_sessions.created_at, EXCLUDED.created_at),
                  last_message  = EXCLUDED.last_message,
                  message_count = EXCLUDED.message_count,
                  updated_at    = now()
            """
            cur = self._cursor()
            cur.execute(q, (chat_id, source, name, created_at_iso, last_message, message_count))
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.conn.rollback()
            raise CustomException(e, sys)

    def upsert_messages(
        self,
        chat_id: str,
        source: str,                  # 'chat' or 'viz'
        messages: List[Dict[str, Any]]
    ) -> int:
        """
        Insert messages (ON CONFLICT DO NOTHING). Returns number inserted.
        Message dict shape accepted from UI:
          { id, sender, text, timestamp, imageUrl?, imageUrls?, imageAlt?, status? }
        """
        if not messages:
            return 0

        rows: List[Tuple] = []
        for m in messages:
            rows.append((
                m.get("id"),
                chat_id,
                source,
                m.get("sender"),
                m.get("text"),
                m.get("imageUrl"),
                m.get("imageUrls") if isinstance(m.get("imageUrls"), list) else None,
                m.get("imageAlt"),
                m.get("status"),
                m.get("timestamp")  # UI stores ISO string; postgres will parse it
            ))

        q = """
        INSERT INTO chat_messages
        (id, chat_id, source, sender, text, image_url, image_urls, image_alt, status, ts)
        VALUES %s
        ON CONFLICT (id) DO NOTHING
        """
        try:
            cur = self._cursor()
            execute_values(cur, q, rows, page_size=500)
            inserted = cur.rowcount if cur.rowcount is not None else 0

            # update message_count with actual total
            cur.execute(
                "UPDATE chat_sessions SET message_count = (SELECT COUNT(*) FROM chat_messages WHERE chat_id = %s), updated_at = now() WHERE id = %s",
                (chat_id, chat_id),
            )
            self.conn.commit()
            cur.close()
            return inserted
        except Exception as e:
            self.conn.rollback()
            raise CustomException(e, sys)

    def save_full_chat_dump(
        self,
        session: Dict[str, Any],
        source: str,  # 'chat' or 'viz'
    ) -> int:
        """
        Convenience: save a whole chat session (metadata + all messages).
        Session dict expected keys (as used by UI):
          id, name, createdAt, lastMessage, messageCount, messages
        Returns number of messages inserted.
        """
        try:
            chat_id = session.get("id")
            if not chat_id:
                raise ValueError("session.id missing")

            self.upsert_chat_session(
                chat_id=chat_id,
                source=source,
                name=session.get("name"),
                created_at_iso=session.get("createdAt"),
                last_message=session.get("lastMessage"),
                message_count=session.get("messageCount"),
            )
            msgs = session.get("messages") or []
            inserted = self.upsert_messages(chat_id, source, msgs)
            return inserted
        except Exception as e:
            raise CustomException(e, sys)

    # ---------- NEW: read helpers used by /chat routes ----------

    def get_chat(self, chat_id: str) -> Dict[str, Any]:
        """
        Fetch a chat row from chat_sessions.
        """
        q = """
        SELECT id, source, name, created_at, last_message, message_count, updated_at
        FROM chat_sessions
        WHERE id = %s
        """
        cur = self._cursor()
        try:
            cur.execute(q, (chat_id,))
            row = cur.fetchone()
            if not row:
                return {}
            keys = ["id", "source", "name", "created_at", "last_message", "message_count", "updated_at"]
            return dict(zip(keys, row))
        finally:
            cur.close()

    def get_messages(
        self,
        chat_id: str,
        source: Optional[str] = None,
        limit: Optional[int] = None,
        asc: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Fetch messages for a chat (optionally filter by source).
        """
        base = """
        SELECT id, chat_id, source, sender, text, image_url, image_urls, image_alt, status, ts
        FROM chat_messages
        WHERE chat_id = %s
        """
        params: List[Any] = [chat_id]
        if source:
            base += " AND source = %s"
            params.append(source)
        order = "ASC" if asc else "DESC"
        base += f" ORDER BY ts {order}"
        if limit and isinstance(limit, int) and limit > 0:
            base += " LIMIT %s"
            params.append(limit)

        cur = self._cursor()
        try:
            cur.execute(base, tuple(params))
            rows = cur.fetchall() or []
            keys = ["id", "chat_id", "source", "sender", "text", "image_url", "image_urls", "image_alt", "status", "ts"]
            return [dict(zip(keys, r)) for r in rows]
        finally:
            cur.close()

    # ------------------ business card contacts ------------------

    @staticmethod
    def _as_list(v):
        if v is None:
            return []
        return v if isinstance(v, list) else [v]

    @staticmethod
    def _addr_parts(a: Any):
        if not a:
            return {}, ""
        if isinstance(a, str):
            return {}, a
        if isinstance(a, dict):
            street = a.get("street")
            city = a.get("city")
            state = a.get("state")
            postal = a.get("postal_code") or a.get("postal") or a.get("zip")
            country = a.get("country")
            parts = {
                "street": street, "city": city, "state": state,
                "postal_code": postal, "country": country
            }
            full = ", ".join([s for s in [street, city, state, postal, country] if s])
            return parts, full
        return {}, str(a)

    def _upsert_flat_for_contact(self, contact_id: str):
        """
        Build/refresh the flat row for one contact_id into business_contacts_flat.
        """
        q = """
        WITH agg AS (
          SELECT
            c.id AS contact_id, c.chat_id, c.source, c.raw_text, c.attachment_urls, c.created_at,
            COALESCE(array_remove(array_agg(DISTINCT n.name),    NULL), '{}') AS names,
            COALESCE(array_remove(array_agg(DISTINCT t.title),   NULL), '{}') AS titles,
            COALESCE(array_remove(array_agg(DISTINCT co.company),NULL), '{}') AS companies,
            COALESCE(array_remove(array_agg(DISTINCT p.phone),   NULL), '{}') AS phones,
            COALESCE(array_remove(array_agg(DISTINCT e.email),   NULL), '{}') AS emails,
            COALESCE(array_remove(array_agg(DISTINCT w.url),     NULL), '{}') AS websites,
            COALESCE(
              array_remove(
                array_agg(DISTINCT COALESCE(a.full_text, concat_ws(', ', a.street, a.city, a.state, a.postal_code, a.country))),
                NULL
              ),
              '{}'
            ) AS addresses
          FROM card_contacts c
          LEFT JOIN card_names      n  ON n.contact_id  = c.id
          LEFT JOIN card_titles     t  ON t.contact_id  = c.id
          LEFT JOIN card_companies  co ON co.contact_id = c.id
          LEFT JOIN card_phones     p  ON p.contact_id  = c.id
          LEFT JOIN card_emails     e  ON e.contact_id  = c.id
          LEFT JOIN card_websites   w  ON w.contact_id  = c.id
          LEFT JOIN card_addresses  a  ON a.contact_id  = c.id
          WHERE c.id = %s
          GROUP BY c.id
        )
        INSERT INTO business_contacts_flat
          (contact_id, chat_id, source, names, titles, companies, phones, emails, websites, addresses, raw_text, attachment_urls, created_at, updated_at)
        SELECT contact_id, chat_id, source, names, titles, companies, phones, emails, websites, addresses, raw_text, attachment_urls, created_at, now()
        FROM agg
        ON CONFLICT (contact_id) DO UPDATE
          SET names = EXCLUDED.names,
              titles = EXCLUDED.titles,
              companies = EXCLUDED.companies,
              phones = EXCLUDED.phones,
              emails = EXCLUDED.emails,
              websites = EXCLUDED.websites,
              addresses = EXCLUDED.addresses,
              raw_text = EXCLUDED.raw_text,
              attachment_urls = EXCLUDED.attachment_urls,
              updated_at = now();
        """
        cur = self._cursor()
        try:
            cur.execute(q, (contact_id,))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def save_business_contacts(
        self,
        contacts: List[Dict[str, Any]],
        chat_id: Optional[str] = None,
        source: str = "viz",                           # 'chat' | 'viz'
        attachment_urls: Optional[List[str]] = None,   # image URLs if you have them
    ) -> List[str]:
        """
        Save N extracted contacts. Supports multi-values & multi-contacts.
        Returns list of inserted contact UUIDs.
        """
        if not contacts:
            return []

        ids: List[str] = []
        cur = self._cursor()
        try:
            for c in contacts:
                raw_text = c.get("raw_text")
                cur.execute(
                    "INSERT INTO card_contacts (chat_id, source, raw_text, attachment_urls) VALUES (%s,%s,%s,%s) RETURNING id",
                    (chat_id, source, raw_text, attachment_urls or [])
                )
                contact_id = cur.fetchone()[0]
                ids.append(contact_id)

                # names
                names = (
                    self._as_list(c.get("names"))
                    or self._as_list(c.get("full_names"))
                    or self._as_list(c.get("other_names"))
                    or self._as_list(c.get("full_name"))
                )
                for name in names:
                    if name:
                        cur.execute("INSERT INTO card_names (contact_id, name) VALUES (%s,%s)", (contact_id, str(name)))

                # titles
                titles = self._as_list(c.get("titles")) or self._as_list(c.get("job_title"))
                for title in titles:
                    if title:
                        cur.execute("INSERT INTO card_titles (contact_id, title) VALUES (%s,%s)", (contact_id, str(title)))

                # companies
                companies = (
                    self._as_list(c.get("organizations"))
                    or self._as_list(c.get("companies"))
                    or self._as_list(c.get("company"))
                    or self._as_list(c.get("organization"))
                )
                for comp in companies:
                    if comp:
                        cur.execute("INSERT INTO card_companies (contact_id, company) VALUES (%s,%s)", (contact_id, str(comp)))

                # phones / emails / websites
                for p in self._as_list(c.get("phones")):
                    if p:
                        cur.execute("INSERT INTO card_phones (contact_id, phone) VALUES (%s,%s)", (contact_id, str(p)))

                for e in self._as_list(c.get("emails")):
                    if e:
                        cur.execute("INSERT INTO card_emails (contact_id, email) VALUES (%s,%s)", (contact_id, str(e)))

                for u in self._as_list(c.get("websites")):
                    if u:
                        cur.execute("INSERT INTO card_websites (contact_id, url) VALUES (%s,%s)", (contact_id, str(u)))

                # addresses (string or object)
                addrs: List[Any] = []
                if c.get("addresses"):
                    addrs.extend(self._as_list(c.get("addresses")))
                if c.get("address"):
                    addrs.append(c.get("address"))
                for a in addrs:
                    parts, full = self._addr_parts(a)
                    cur.execute(
                        """
                        INSERT INTO card_addresses
                          (contact_id, street, city, state, postal_code, country, full_text)
                        VALUES (%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            contact_id,
                            parts.get("street"),
                            parts.get("city"),
                            parts.get("state"),
                            parts.get("postal_code"),
                            parts.get("country"),
                            full or None,
                        ),
                    )

                # sync flat table for this contact
                self._upsert_flat_for_contact(contact_id)

            self.conn.commit()
            return ids
        except Exception as e:
            self.conn.rollback()
            raise CustomException(e, sys)
        finally:
            cur.close()

    # ---------- NEW: flat list + rebuild + Excel export ----------

    def list_contacts_flat(
        self,
        chat_id: str,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return rows from business_contacts_flat for a chat (optionally filter by source).
        """
        q = """
        SELECT contact_id, chat_id, source, names, titles, companies, phones, emails, websites, addresses,
               raw_text, attachment_urls, created_at, updated_at
        FROM business_contacts_flat
        WHERE chat_id = %s
        """
        params: List[Any] = [chat_id]
        if source:
            q += " AND source = %s"
            params.append(source)
        q += " ORDER BY created_at DESC"

        cur = self._cursor()
        try:
            cur.execute(q, tuple(params))
            rows = cur.fetchall() or []
            cols = [
                "contact_id", "chat_id", "source", "names", "titles", "companies",
                "phones", "emails", "websites", "addresses",
                "raw_text", "attachment_urls", "created_at", "updated_at"
            ]
            return [dict(zip(cols, r)) for r in rows]
        finally:
            cur.close()

    def rebuild_flat_for_chat(self, chat_id: str) -> int:
        """
        Recompute flat rows for all contacts in a chat.
        """
        cur = self._cursor()
        try:
            cur.execute("SELECT id FROM card_contacts WHERE chat_id = %s", (chat_id,))
            ids = [r[0] for r in (cur.fetchall() or [])]
            for cid in ids:
                self._upsert_flat_for_contact(cid)
            self.conn.commit()
            return len(ids)
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def export_contacts_to_excel(
        self,
        chat_id: str,
        source: Optional[str] = None,
        out_dir: str = "static/exports",
        filename_prefix: str = "business_cards",
    ) -> Dict[str, str]:
        """
        Build a clean Excel for all contacts in a chat and return:
          { "file_path": "<abs>", "api_path": "/static/exports/<file>" }
        """
        # get flat rows (rebuild if empty)
        rows = self.list_contacts_flat(chat_id, source=source)
        if not rows:
            self.rebuild_flat_for_chat(chat_id)
            rows = self.list_contacts_flat(chat_id, source=source)

        if not rows:
            raise CustomException(f"No contacts found for chat_id={chat_id}", sys)

        def join_list(x: Any) -> str:
            if x is None:
                return ""
            if isinstance(x, list):
                return ", ".join([str(v) for v in x if v is not None and str(v).strip() != ""])
            return str(x or "")

        data = []
        for r in rows:
            name = (r.get("names") or [])
            title = (r.get("titles") or [])
            company = (r.get("companies") or [])
            addr = (r.get("addresses") or [])

            data.append({
                "Full Name": name[0] if name else "",
                "Company": company[0] if company else "",
                "Title": title[0] if title else "",
                "Phones": join_list(r.get("phones")),
                "Emails": join_list(r.get("emails")),
                "Websites": join_list(r.get("websites")),
                "Address": addr[0] if addr else "",
                "Source": r.get("source") or "",
                "Chat ID": r.get("chat_id") or "",
                "Created At": r.get("created_at"),
            })

        df = pd.DataFrame(data, columns=[
            "Full Name", "Company", "Title", "Phones", "Emails", "Websites",
            "Address", "Source", "Chat ID", "Created At"
        ])

        # ensure directory
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        file_name = f"{filename_prefix}_{chat_id}_{stamp}.xlsx"
        abs_path = os.path.abspath(os.path.join(out_dir, file_name))
        api_path = f"/{out_dir.strip('/')}" + f"/{file_name}"

        # write with XlsxWriter if available; fallback to openpyxl
        engine = "xlsxwriter"
        try:
            with pd.ExcelWriter(abs_path, engine=engine) as writer:
                df.to_excel(writer, index=False, sheet_name="Contacts")
        except Exception:
            # fallback
            with pd.ExcelWriter(abs_path, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Contacts")

        logging.info(f"📤 Exported {len(df)} contacts to {abs_path}")
        return {"file_path": abs_path, "api_path": api_path}

    # ------------------ close ------------------

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
