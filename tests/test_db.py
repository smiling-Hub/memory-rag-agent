# -*- coding: utf-8 -*-
"""db.py 的单元测试：验证注册/登录/充值/扣费这些「碰钱」逻辑的正确性。

跑法：在 loyal-elephie-demo 目录下  venv/Scripts/python -m pytest tests/ -v
"""
import os
import tempfile
import db


def setup_module():
    """用临时文件隔离测试，不污染真实的 app.db"""
    db.DB_PATH = os.path.join(tempfile.gettempdir(), "test_loyal_elephie.db")
    if os.path.exists(db.DB_PATH):
        os.remove(db.DB_PATH)
    db.init_db()


def teardown_module():
    if os.path.exists(db.DB_PATH):
        os.remove(db.DB_PATH)


def test_register_and_login():
    assert db.register("alice", "secret123") is True
    u = db.login("alice", "secret123")
    assert u is not None
    assert u["username"] == "alice"
    # 关键：库里存的必须是哈希，不是明文密码
    assert u["password_hash"] != "secret123"


def test_register_duplicate_fails():
    db.register("bob", "pass")
    assert db.register("bob", "pass") is False


def test_login_wrong_password_fails():
    db.register("carol", "right")
    assert db.login("carol", "wrong") is None


def test_add_credits_and_balance():
    db.register("dave", "p", initial_credits=50)
    dave = db.login("dave", "p")
    assert db.get_balance(dave["id"]) == 50
    db.add_credits(dave["id"], 30)
    assert db.get_balance(dave["id"]) == 80


def test_charge_deducts():
    """250 token → 费率 100/credit → 向上取整 3 credits"""
    db.register("eve", "p", initial_credits=100)
    eve = db.login("eve", "p")
    cost = db.charge(eve["id"], 250, note="test")
    assert cost == 3
    assert db.get_balance(eve["id"]) == 97


def test_charge_insufficient_raises():
    """余额不足要抛异常，不能静默扣成负数"""
    db.register("frank", "p", initial_credits=1)
    frank = db.login("frank", "p")
    db.charge(frank["id"], 10)  # 扣 1，余额归 0
    try:
        db.charge(frank["id"], 10)  # 再扣应失败
        assert False, "余额不足应该抛 ValueError"
    except ValueError:
        pass


def test_charge_never_negative():
    """扣费失败时余额应保持不变"""
    db.register("grace", "p", initial_credits=2)
    grace = db.login("grace", "p")
    try:
        db.charge(grace["id"], 1000)  # cost=10 > 2，失败
    except ValueError:
        pass
    assert db.get_balance(grace["id"]) == 2  # 余额没变
