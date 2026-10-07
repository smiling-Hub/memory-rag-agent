# -*- coding: utf-8 -*-
"""用户与 Credits 计费的 SQL 层（sqlite3 + bcrypt）。"""
import sqlite3
import bcrypt

DB_PATH = "app.db"
CREDIT_RATE = 100  # 计费费率：每 100 个 token 收 1 credit


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row  # 让查询结果能按列名取（row["username"]）
    return c


def init_db():
    """建表：users（用户+余额）、usage_log（计费流水）"""
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                credits INTEGER NOT NULL DEFAULT 0
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                tokens INTEGER NOT NULL,
                cost INTEGER NOT NULL,
                note TEXT,
                created_at TEXT DEFAULT (datetime('now','localtime'))
            )
        """)


def register(username, password, initial_credits=0):
    """注册：密码哈希后存库。返回 True/False（False 表示用户名已存在）"""
    h = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        with _conn() as c:
            c.execute("INSERT INTO users (username, password_hash, credits) VALUES (?,?,?)",
                      (username, h, initial_credits))
        return True
    except sqlite3.IntegrityError:
        return False


def login(username, password):
    """登录：拿 bcrypt 比对哈希，成功返回用户行，失败返回 None"""
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if row and bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return dict(row)
    return None


def get_balance(user_id):
    with _conn() as c:
        row = c.execute("SELECT credits FROM users WHERE id=?", (user_id,)).fetchone()
    return row["credits"]


def add_credits(user_id, amount):
    """充值"""
    with _conn() as c:
        c.execute("UPDATE users SET credits = credits + ? WHERE id=?", (amount, user_id))


def charge(user_id, tokens, note=""):
    """按 token 扣费。关键：一条 SQL 里同时检查余额并扣减，保证原子性（不会超扣）。
    余额不足抛 ValueError。返回本次扣了多少 credit。"""
    cost = max(1, -(-tokens // CREDIT_RATE))  # 向上取整，最低扣 1
    with _conn() as c:
        cur = c.execute(
            "UPDATE users SET credits = credits - ? WHERE id=? AND credits >= ?",
            (cost, user_id, cost),
        )
        if cur.rowcount == 0:
            raise ValueError("余额不足，请先充值")
        c.execute("INSERT INTO usage_log (user_id, tokens, cost, note) VALUES (?,?,?,?)",
                  (user_id, tokens, cost, note))
    return cost
