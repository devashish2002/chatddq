# db.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.types import JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import json
from typing import Optional, List, Dict, Any

Base = declarative_base()

class Assessment(Base):
    __tablename__ = "assessments"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    topic = Column(String, index=True)
    score = Column(Float)
    total_questions = Column(Integer)
    correct_answers = Column(Integer)
    difficulty_distribution = Column(JSON)
    subtopic_performance = Column(JSON)
    weaknesses = Column(JSON)
    strengths = Column(JSON)
    recommended_level = Column(String)
    questions = Column(JSON)   # list of question dicts
    answers = Column(JSON)     # list of user answer indices
    raw = Column(Text)         # optional full JSON dump
    created_at = Column(DateTime, default=datetime.utcnow)

# DB engine & session
ENGINE_STR = "sqlite:///assessments.db"
engine = create_engine(ENGINE_STR, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# create tables
Base.metadata.create_all(bind=engine)


def save_assessment_result(
    user_id: str,
    topic: str,
    assessment_result,
    questions: Optional[List[Dict]] = None,
    answers: Optional[List[int]] = None,
) -> int:
    """
    Save an assessment result into the DB.
    - assessment_result: dataclass-like object or dict (should contain score, total_questions, correct_answers, ...)
    - questions: optional list of question dicts (so UI can later review them)
    - answers: optional list of user answer indices (so UI can show which options user picked)
    Returns the DB row id.
    """
    session = SessionLocal()
    try:
        # Coerce assessment_result to a dict-like structure
        if hasattr(assessment_result, "__dict__"):
            ar = dict(vars(assessment_result))
        elif isinstance(assessment_result, dict):
            ar = dict(assessment_result)
        else:
            # fallback: pull typical attributes, if present
            ar = {
                "score": getattr(assessment_result, "score", None),
                "total_questions": getattr(assessment_result, "total_questions", None),
                "correct_answers": getattr(assessment_result, "correct_answers", None),
                "difficulty_distribution": getattr(assessment_result, "difficulty_distribution", None),
                "subtopic_performance": getattr(assessment_result, "subtopic_performance", None),
                "weaknesses": getattr(assessment_result, "weaknesses", None),
                "strengths": getattr(assessment_result, "strengths", None),
                "recommended_level": getattr(assessment_result, "recommended_level", None),
            }

        qlist = []
        if questions:
            qlist = [q.__dict__ if hasattr(q, "__dict__") else q for q in questions]
        elif ar.get("questions"):
            qlist = ar.get("questions")

        alist = answers or ar.get("answers") or []

        row = Assessment(
            user_id=user_id,
            topic=topic,
            score=float(ar.get("score") or 0.0),
            total_questions=int(ar.get("total_questions") or (len(qlist) or 0)),
            correct_answers=int(ar.get("correct_answers") or 0),
            difficulty_distribution=ar.get("difficulty_distribution") or {},
            subtopic_performance=ar.get("subtopic_performance") or {},
            weaknesses=ar.get("weaknesses") or [],
            strengths=ar.get("strengths") or [],
            recommended_level=ar.get("recommended_level") or "",
            questions=qlist,
            answers=alist,
            raw=json.dumps(ar, default=str),
            created_at=datetime.utcnow()
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id
    finally:
        session.close()


def load_assessment_results(user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load all (or up to `limit`) assessments for a user.
    Returns a list of dicts in chronological order (oldest first) so callers can use [-1] for latest.
    """
    session = SessionLocal()
    try:
        q = session.query(Assessment).filter(Assessment.user_id == user_id).order_by(Assessment.created_at.asc())
        if limit:
            rows = q.limit(limit).all()
        else:
            rows = q.all()

        out = []
        for r in rows:
            out.append({
                "id": r.id,
                "user_id": r.user_id,
                "topic": r.topic,
                "score": r.score,
                "total_questions": r.total_questions,
                "correct_answers": r.correct_answers,
                "difficulty_distribution": r.difficulty_distribution,
                "subtopic_performance": r.subtopic_performance,
                "weaknesses": r.weaknesses or [],
                "strengths": r.strengths or [],
                "recommended_level": r.recommended_level,
                "questions": r.questions or [],
                "answers": r.answers or [],
                "timestamp": r.created_at.isoformat()
            })
        return out
    finally:
        session.close()


def get_recent_assessments_text(user_id: str, topic: Optional[str] = None, limit: int = 5) -> str:
    """
    Produce a short textual summary (most recent `limit`) of a user's assessments,
    useful to include in LLM prompts.
    """
    session = SessionLocal()
    try:
        q = session.query(Assessment).filter(Assessment.user_id == user_id)
        if topic:
            q = q.filter(Assessment.topic == topic)
        rows = q.order_by(Assessment.created_at.desc()).limit(limit).all()  # newest first
        if not rows:
            return "No prior assessments found."

        parts = []
        for r in rows:
            strengths = ", ".join(r.strengths or []) if r.strengths else "None"
            weaknesses = ", ".join(r.weaknesses or []) if r.weaknesses else "None"
            parts.append(
                f"[{r.created_at.date()}] topic={r.topic} score={r.score:.2f} level={r.recommended_level} "
                f"strengths={strengths} weaknesses={weaknesses}"
            )
        # return newest → oldest as text (you can reverse if you prefer oldest first)
        return "\n".join(parts)
    finally:
        session.close()
