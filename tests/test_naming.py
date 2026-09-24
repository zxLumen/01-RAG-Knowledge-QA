import pytest

from src.vectorstore import naming


def test_storage_name_ascii_identity():
    assert naming.storage_name("default") == "default"


def test_storage_name_non_ascii_legal():
    name = naming.storage_name("python文档")
    assert naming._STORAGE_RE.match(name)


def test_alias_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(naming, "_ALIAS_FILE", tmp_path / "aliases.json")
    storage = naming.storage_name("python文档")
    naming.register_alias("python文档", storage)
    assert naming.display_name(storage) == "python文档"
    assert naming.storage_for_display("python文档") == storage
    assert naming.storage_for_display(storage) == storage


def test_rename_display(tmp_path, monkeypatch):
    monkeypatch.setattr(naming, "_ALIAS_FILE", tmp_path / "aliases.json")
    storage = naming.storage_name("旧名")
    naming.register_alias("旧名", storage)
    naming.rename_display("旧名", "新名")
    assert naming.display_name(storage) == "新名"
    assert naming.storage_for_display("新名") == storage
    assert naming.storage_for_display("旧名") == "旧名"
    assert naming.storage_for_display(storage) == storage


def test_rename_display_no_json(tmp_path, monkeypatch):
    monkeypatch.setattr(naming, "_ALIAS_FILE", tmp_path / "aliases.json")
    naming.rename_display("default", "主库")
    assert naming.display_name("default") == "主库"
    assert naming.storage_for_display("主库") == "default"


def test_rename_display_to_identity_clears_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(naming, "_ALIAS_FILE", tmp_path / "aliases.json")
    storage = naming.storage_name("旧名")
    naming.register_alias("旧名", storage)
    naming.rename_display("旧名", storage)
    assert naming.display_name(storage) == storage


def test_rename_display_blank_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(naming, "_ALIAS_FILE", tmp_path / "aliases.json")
    storage = naming.storage_name("旧名")
    naming.register_alias("旧名", storage)
    with pytest.raises(ValueError):
        naming.rename_display("旧名", "   ")
    assert naming.display_name(storage) == "旧名"


def test_rename_display_replaces_old_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(naming, "_ALIAS_FILE", tmp_path / "aliases.json")
    storage = naming.storage_name("a")
    naming.register_alias("a2", storage)
    assert naming.display_name(storage) == "a2"
    naming.rename_display("a2", "a3")
    assert naming.display_name(storage) == "a3"


def test_delete_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(naming, "_ALIAS_FILE", tmp_path / "aliases.json")
    storage = naming.storage_name("猫")
    naming.register_alias("猫", storage)
    naming.register_alias("cat", storage)
    assert naming.display_name(storage) == "猫"
    naming.delete_alias(storage)
    assert naming.display_name(storage) == storage
    assert naming.storage_for_display("猫") == "猫"


def test_creation_order_newest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(naming, "_ORDER_FILE", tmp_path / "order.json")
    naming.record_creation("a")
    naming.record_creation("b")
    naming.record_creation("c")
    naming.record_creation("b")  # idempotent, keeps original position
    assert naming.order_index() == {"a": 0, "b": 1, "c": 2}


def test_drop_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(naming, "_ORDER_FILE", tmp_path / "order.json")
    naming.record_creation("a")
    naming.record_creation("b")
    naming.drop_creation("a")
    assert naming.order_index() == {"b": 0}

