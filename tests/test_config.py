from src.config import get_settings


def test_db_url_accepts_database_url(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.delenv("DB_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://example.com:5432/app")

    settings = get_settings()

    assert settings.db_url == "postgresql://example.com:5432/app"
    get_settings.cache_clear()
