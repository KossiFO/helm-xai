import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from helm import UCB1Learner, UserProfile, __version__


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS explanations (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, model TEXT NOT NULL,
                    profile TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    explanation_id TEXT NOT NULL REFERENCES explanations(id),
                    method TEXT NOT NULL, rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                    updated_at TEXT NOT NULL, PRIMARY KEY(explanation_id, method)
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, result):
        with self.connect() as db:
            db.execute("INSERT INTO explanations VALUES (?, ?, ?, ?, ?)",
                       (result["id"], result["created_at"], result["model"], result["profile"],
                        json.dumps(result, ensure_ascii=False, allow_nan=False)))

    def get(self, explanation_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM explanations WHERE id=?", (explanation_id,)).fetchone()
            if row is None:
                raise KeyError(explanation_id)
            result = json.loads(row["payload"])
            result["feedback"] = {r["method"]: r["rating"] for r in db.execute(
                "SELECT method, rating FROM feedback WHERE explanation_id=?", (explanation_id,))}
            return result

    def history(self):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM explanations ORDER BY created_at DESC LIMIT 50").fetchall()
        results = []
        for row in rows:
            r = json.loads(row["payload"])
            results.append({k: r[k] for k in ["id", "created_at", "profile", "model", "text", "prediction"]})
        return results

    def rate(self, explanation_id, method, rating):
        result = self.get(explanation_id)
        if method not in result["attributions"]:
            raise ValueError("Cette méthode n'a pas produit d'explication pour cette analyse.")
        with self.connect() as db:
            db.execute("""INSERT INTO feedback VALUES (?, ?, ?, ?)
                          ON CONFLICT(explanation_id, method) DO UPDATE
                          SET rating=excluded.rating, updated_at=excluded.updated_at""",
                       (explanation_id, method, rating, now()))
        return result

    def learner(self, model, package_version=__version__):
        learner = UCB1Learner()
        with self.connect() as db:
            rows = db.execute("""SELECT e.profile, e.payload, f.method, f.rating FROM feedback f
                                  JOIN explanations e ON e.id=f.explanation_id WHERE e.model=?
                                  ORDER BY e.created_at, f.method""", (model,)).fetchall()
        for row in rows:
            if json.loads(row["payload"]).get("package_version") != package_version:
                continue
            learner.record(UserProfile(row["profile"]), row["method"], row["rating"])
        return learner
