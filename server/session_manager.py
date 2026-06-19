import os
import uuid
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALERTS_DIR = os.path.join(BASE_DIR, "alerts")
DB_PATH = os.path.join(BASE_DIR, "proctor.db")

os.makedirs(ALERTS_DIR, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class AlertType(str, Enum):
    NO_FACE = "no_face"
    GAZE_DEVIATION = "gaze_deviation"
    MULTI_FACE = "multi_face"
    FACE_REPLACEMENT = "face_replacement"
    FACE_OFF_CENTER = "face_off_center"


class AlertStatus(str, Enum):
    PENDING = "pending"
    FALSE_POSITIVE = "false_positive"
    CONFIRMED_CHEATING = "confirmed_cheating"


class ExamStatus(str, Enum):
    NOT_STARTED = "not_started"
    VERIFYING = "verifying"
    IN_PROGRESS = "in_progress"
    ENDED = "ended"
    TERMINATED = "terminated"


class Student(Base):
    __tablename__ = "students"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    student_no = Column(String, unique=True, index=True)
    face_encoding = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    exams = relationship("ExamSession", back_populates="student")


class ExamSession(Base):
    __tablename__ = "exam_sessions"

    id = Column(String, primary_key=True, index=True)
    student_id = Column(String, ForeignKey("students.id"))
    exam_name = Column(String)
    status = Column(String, default=ExamStatus.NOT_STARTED.value)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    verify_confidence = Column(Float, default=0.0)
    total_alerts = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="exams")
    alerts = relationship("AlertEvent", back_populates="exam_session")


class AlertEvent(Base):
    __tablename__ = "alert_events"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("exam_sessions.id"))
    alert_type = Column(String, index=True)
    confidence = Column(Float, default=0.0)
    screenshot_path = Column(String, nullable=True)
    status = Column(String, default=AlertStatus.PENDING.value)
    timestamp = Column(DateTime, default=datetime.utcnow)
    description = Column(Text, nullable=True)

    exam_session = relationship("ExamSession", back_populates="alerts")


Base.metadata.create_all(bind=engine)


@dataclass
class SessionState:
    session_id: str
    student_id: str
    last_face_encoding: Optional[np.ndarray] = None
    no_face_start_time: float = 0.0
    gaze_deviation_start_time: float = 0.0
    last_frame_time: float = 0.0
    alert_buffer: List[Dict] = field(default_factory=list)
    sse_clients: List = field(default_factory=list)


class SessionManager:
    def __init__(self):
        self._active_sessions: Dict[str, SessionState] = {}
        self._sse_callbacks: Dict[str, List] = {}

    def create_student(self, name: str, student_no: str) -> Student:
        db = SessionLocal()
        try:
            existing = db.query(Student).filter(Student.student_no == student_no).first()
            if existing:
                return existing
            student = Student(
                id=str(uuid.uuid4()),
                name=name,
                student_no=student_no
            )
            db.add(student)
            db.commit()
            db.refresh(student)
            return student
        finally:
            db.close()

    def set_student_face(self, student_id: str, face_encoding: np.ndarray) -> bool:
        db = SessionLocal()
        try:
            student = db.query(Student).filter(Student.id == student_id).first()
            if not student:
                return False
            student.face_encoding = json.dumps(face_encoding.tolist())
            db.commit()
            return True
        finally:
            db.close()

    def get_student_face(self, student_id: str) -> Optional[np.ndarray]:
        db = SessionLocal()
        try:
            student = db.query(Student).filter(Student.id == student_id).first()
            if not student or not student.face_encoding:
                return None
            return np.array(json.loads(student.face_encoding))
        finally:
            db.close()

    def create_exam_session(self, student_id: str, exam_name: str) -> ExamSession:
        db = SessionLocal()
        try:
            session = ExamSession(
                id=str(uuid.uuid4()),
                student_id=student_id,
                exam_name=exam_name,
                status=ExamStatus.NOT_STARTED.value
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return session
        finally:
            db.close()

    def start_exam(self, session_id: str, verify_confidence: float = 0.0) -> bool:
        db = SessionLocal()
        try:
            session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
            if not session:
                return False
            session.status = ExamStatus.IN_PROGRESS.value
            session.start_time = datetime.utcnow()
            session.verify_confidence = verify_confidence
            db.commit()

            self._active_sessions[session_id] = SessionState(
                session_id=session_id,
                student_id=session.student_id
            )
            return True
        finally:
            db.close()

    def end_exam(self, session_id: str, terminated: bool = False) -> bool:
        db = SessionLocal()
        try:
            session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
            if not session:
                return False
            session.status = ExamStatus.TERMINATED.value if terminated else ExamStatus.ENDED.value
            session.end_time = datetime.utcnow()
            db.commit()

            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
            return True
        finally:
            db.close()

    def get_session(self, session_id: str) -> Optional[ExamSession]:
        db = SessionLocal()
        try:
            return db.query(ExamSession).filter(ExamSession.id == session_id).first()
        finally:
            db.close()

    def get_session_state(self, session_id: str) -> Optional[SessionState]:
        return self._active_sessions.get(session_id)

    def add_alert(self, session_id: str, alert_type: AlertType, confidence: float,
                  screenshot_path: Optional[str] = None, description: Optional[str] = None) -> AlertEvent:
        db = SessionLocal()
        try:
            alert = AlertEvent(
                id=str(uuid.uuid4()),
                session_id=session_id,
                alert_type=alert_type.value,
                confidence=confidence,
                screenshot_path=screenshot_path,
                status=AlertStatus.PENDING.value,
                description=description or ""
            )
            db.add(alert)

            session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
            if session:
                session.total_alerts += 1

            db.commit()
            db.refresh(alert)

            self._push_sse_alert(session_id, alert)
            return alert
        finally:
            db.close()

    def update_alert_status(self, alert_id: str, status: AlertStatus) -> bool:
        db = SessionLocal()
        try:
            alert = db.query(AlertEvent).filter(AlertEvent.id == alert_id).first()
            if not alert:
                return False
            alert.status = status.value

            if status == AlertStatus.CONFIRMED_CHEATING:
                session = db.query(ExamSession).filter(ExamSession.id == alert.session_id).first()
                if session and session.status == ExamStatus.IN_PROGRESS.value:
                    session.status = ExamStatus.TERMINATED.value
                    session.end_time = datetime.utcnow()
                    if alert.session_id in self._active_sessions:
                        del self._active_sessions[alert.session_id]

            db.commit()
            return True
        finally:
            db.close()

    def get_alerts(self, session_id: str, status: Optional[AlertStatus] = None) -> List[AlertEvent]:
        db = SessionLocal()
        try:
            query = db.query(AlertEvent).filter(AlertEvent.session_id == session_id)
            if status:
                query = query.filter(AlertEvent.status == status.value)
            return query.order_by(AlertEvent.timestamp.desc()).all()
        finally:
            db.close()

    def get_all_sessions(self) -> List[ExamSession]:
        db = SessionLocal()
        try:
            return db.query(ExamSession).order_by(ExamSession.created_at.desc()).all()
        finally:
            db.close()

    def generate_report(self, session_id: str) -> Dict:
        db = SessionLocal()
        try:
            session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
            if not session:
                return {}

            alerts = db.query(AlertEvent).filter(AlertEvent.session_id == session_id).order_by(
                AlertEvent.timestamp.asc()
            ).all()

            alert_stats = {}
            for alert in alerts:
                atype = alert.alert_type
                if atype not in alert_stats:
                    alert_stats[atype] = {"count": 0, "avg_confidence": 0.0, "confirmed": 0, "false_positive": 0}
                alert_stats[atype]["count"] += 1
                alert_stats[atype]["avg_confidence"] += alert.confidence
                if alert.status == AlertStatus.CONFIRMED_CHEATING.value:
                    alert_stats[atype]["confirmed"] += 1
                elif alert.status == AlertStatus.FALSE_POSITIVE.value:
                    alert_stats[atype]["false_positive"] += 1

            for k in alert_stats:
                if alert_stats[k]["count"] > 0:
                    alert_stats[k]["avg_confidence"] = round(
                        alert_stats[k]["avg_confidence"] / alert_stats[k]["count"], 4
                    )

            timeline = []
            for alert in alerts:
                timeline.append({
                    "id": alert.id,
                    "type": alert.alert_type,
                    "confidence": alert.confidence,
                    "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
                    "screenshot": alert.screenshot_path,
                    "status": alert.status,
                    "description": alert.description
                })

            overall_confidence = 0.0
            if alert_stats:
                confs = [v["avg_confidence"] for v in alert_stats.values()]
                overall_confidence = sum(confs) / len(confs) if confs else 0.0

            duration_seconds = 0
            if session.start_time and session.end_time:
                duration_seconds = int((session.end_time - session.start_time).total_seconds())
            elif session.start_time:
                duration_seconds = int((datetime.utcnow() - session.start_time).total_seconds())

            return {
                "session_id": session.id,
                "student_id": session.student_id,
                "exam_name": session.exam_name,
                "status": session.status,
                "start_time": session.start_time.isoformat() if session.start_time else None,
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "duration_seconds": duration_seconds,
                "verify_confidence": session.verify_confidence,
                "total_alerts": session.total_alerts,
                "alert_stats": alert_stats,
                "overall_confidence_score": round(overall_confidence, 4),
                "timeline": timeline
            }
        finally:
            db.close()

    def register_sse_callback(self, session_id: str, callback) -> None:
        if session_id not in self._sse_callbacks:
            self._sse_callbacks[session_id] = []
        self._sse_callbacks[session_id].append(callback)

    def unregister_sse_callback(self, session_id: str, callback) -> None:
        if session_id in self._sse_callbacks:
            try:
                self._sse_callbacks[session_id].remove(callback)
            except ValueError:
                pass

    def _push_sse_alert(self, session_id: str, alert: AlertEvent) -> None:
        callbacks = self._sse_callbacks.get(session_id, [])
        alert_data = {
            "id": alert.id,
            "type": alert.alert_type,
            "confidence": alert.confidence,
            "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
            "screenshot": alert.screenshot_path,
            "status": alert.status,
            "description": alert.description
        }
        for cb in callbacks:
            try:
                cb(alert_data)
            except Exception:
                pass

    def save_screenshot(self, session_id: str, frame, alert_type: str) -> str:
        import cv2
        timestamp = int(time.time() * 1000)
        filename = f"{session_id}_{alert_type}_{timestamp}.jpg"
        filepath = os.path.join(ALERTS_DIR, filename)
        cv2.imwrite(filepath, frame)
        return f"/alerts/{filename}"


session_manager = SessionManager()
