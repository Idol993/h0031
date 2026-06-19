import os
import uuid
import json
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Boolean, Text, ForeignKey, or_, and_
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


class VideoStatus(str, Enum):
    NONE = "none"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"


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
    video_clip_path = Column(String, nullable=True)
    video_status = Column(String, nullable=True)
    status = Column(String, default=AlertStatus.PENDING.value)
    timestamp = Column(DateTime, default=datetime.utcnow)
    description = Column(Text, nullable=True)

    exam_session = relationship("ExamSession", back_populates="alerts")


def _pragmas_to_dict(rows):
    return {row[1]: row for row in rows}


def _migrate_database():
    with engine.connect() as conn:
        result = conn.exec_driver_sql("PRAGMA table_info(alert_events)").fetchall()
        cols = _pragmas_to_dict(result)
        existing = set(cols.keys())

        if "video_clip_path" not in existing:
            try:
                conn.exec_driver_sql(
                    "ALTER TABLE alert_events ADD COLUMN video_clip_path TEXT"
                )
                conn.commit()
            except Exception:
                pass

        if "video_status" not in existing:
            try:
                conn.exec_driver_sql(
                    "ALTER TABLE alert_events ADD COLUMN video_status TEXT"
                )
                conn.commit()
            except Exception:
                pass

        if "video_status" in existing or True:
            try:
                conn.exec_driver_sql(
                    "UPDATE alert_events SET video_status = 'ready' WHERE video_status IS NULL AND video_clip_path IS NOT NULL AND video_clip_path != ''"
                )
                conn.commit()
            except Exception:
                pass
            try:
                conn.exec_driver_sql(
                    "UPDATE alert_events SET video_status = 'none' WHERE video_status IS NULL OR video_status = ''"
                )
                conn.commit()
            except Exception:
                pass


Base.metadata.create_all(bind=engine)
_migrate_database()


@dataclass
class FrameRecord:
    frame: np.ndarray
    timestamp: float


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
    frame_ring_buffer: deque = field(default_factory=lambda: deque(maxlen=150))
    post_alert_recordings: List[Dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


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
                self.flush_all_pending_recordings(session_id)
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

    def is_exam_active(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        if not session:
            return False
        return session.status == ExamStatus.IN_PROGRESS.value

    def push_frame_to_buffer(self, session_id: str, frame: np.ndarray) -> None:
        state = self._active_sessions.get(session_id)
        if not state:
            return
        with state.lock:
            rec = FrameRecord(frame=frame.copy(), timestamp=time.time())
            state.frame_ring_buffer.append(rec)
            for i in range(len(state.post_alert_recordings) - 1, -1, -1):
                recording = state.post_alert_recordings[i]
                recording["frames"].append(FrameRecord(frame=frame.copy(), timestamp=time.time()))
                elapsed = time.time() - recording["start_time"]
                if elapsed >= recording["duration"]:
                    self._flush_single_alert_clip_locked(session_id, state, recording)
                    state.post_alert_recordings.pop(i)

    def flush_all_pending_recordings(self, session_id: str) -> None:
        state = self._active_sessions.get(session_id)
        if not state:
            return
        with state.lock:
            for recording in list(state.post_alert_recordings):
                try:
                    self._flush_single_alert_clip_locked(session_id, state, recording)
                except Exception:
                    pass
            state.post_alert_recordings.clear()

    def _flush_single_alert_clip_locked(self, session_id: str, state: SessionState, recording: Dict) -> None:
        all_frames = recording.get("pre_frames", []) + recording["frames"]
        video_path = None
        video_status = VideoStatus.FAILED.value
        if all_frames:
            try:
                video_path = self._write_video_clip(
                    session_id, all_frames, recording["alert_type"], recording["alert_ts"]
                )
                if video_path:
                    video_status = VideoStatus.READY.value
            except Exception:
                video_path = None
                video_status = VideoStatus.FAILED.value
        else:
            video_status = VideoStatus.NONE.value

        if recording.get("alert_id"):
            db = SessionLocal()
            try:
                alert = db.query(AlertEvent).filter(AlertEvent.id == recording["alert_id"]).first()
                if alert:
                    if video_path:
                        alert.video_clip_path = video_path
                    alert.video_status = video_status
                    db.commit()
            finally:
                db.close()

    def _write_video_clip(self, session_id: str, frames: List[FrameRecord],
                          alert_type: str, alert_ts: int) -> Optional[str]:
        import cv2
        if not frames:
            return None

        filename = f"{session_id}_{alert_type}_{alert_ts}.avi"
        filepath = os.path.join(ALERTS_DIR, filename)

        h, w = frames[0].frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 10.0
        if len(frames) > 1:
            duration = frames[-1].timestamp - frames[0].timestamp
            if duration > 0:
                fps = len(frames) / duration
                fps = max(5.0, min(30.0, fps))

        writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
        if not writer.isOpened():
            return None
        for rec in frames:
            writer.write(rec.frame)
        writer.release()

        return f"/alerts/{filename}"

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
                video_clip_path=None,
                video_status=VideoStatus.GENERATING.value,
                status=AlertStatus.PENDING.value,
                description=description or ""
            )
            db.add(alert)

            session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
            if session:
                session.total_alerts += 1

            db.commit()
            db.refresh(alert)

            state = self._active_sessions.get(session_id)
            if state:
                with state.lock:
                    pre_frames = list(state.frame_ring_buffer)
                    recording = {
                        "alert_id": alert.id,
                        "alert_type": alert_type.value,
                        "alert_ts": int(time.time() * 1000),
                        "start_time": time.time(),
                        "duration": 3.0,
                        "pre_frames": pre_frames,
                        "frames": []
                    }
                    state.post_alert_recordings.append(recording)

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
                    sid = alert.session_id
                    db.commit()
                    if sid in self._active_sessions:
                        self.flush_all_pending_recordings(sid)
                        del self._active_sessions[sid]
                    return True

            db.commit()
            return True
        finally:
            db.close()

    def _normalize_video_status(self, obj: AlertEvent) -> str:
        vs = obj.video_status
        if vs:
            return vs
        if obj.video_clip_path:
            return VideoStatus.READY.value
        return VideoStatus.NONE.value

    def get_alerts(self, session_id: str, status: Optional[AlertStatus] = None) -> List[AlertEvent]:
        db = SessionLocal()
        try:
            query = db.query(AlertEvent).filter(AlertEvent.session_id == session_id)
            if status:
                query = query.filter(AlertEvent.status == status.value)
            result = query.order_by(AlertEvent.timestamp.desc()).all()
            for r in result:
                r.video_status = self._normalize_video_status(r)
            return result
        finally:
            db.close()

    def get_all_sessions(self) -> List[ExamSession]:
        db = SessionLocal()
        try:
            return db.query(ExamSession).order_by(ExamSession.created_at.desc()).all()
        finally:
            db.close()

    def query_alerts_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        session_id: Optional[str] = None,
        student_id: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        status: Optional[AlertStatus] = None,
        exam_name: Optional[str] = None
    ) -> Dict:
        db = SessionLocal()
        try:
            query = db.query(AlertEvent).join(AlertEvent.exam_session).join(ExamSession.student)

            if session_id:
                query = query.filter(AlertEvent.session_id == session_id)
            if student_id:
                query = query.filter(ExamSession.student_id == student_id)
            if alert_type:
                query = query.filter(AlertEvent.alert_type == alert_type.value)
            if status:
                query = query.filter(AlertEvent.status == status.value)
            if exam_name:
                query = query.filter(ExamSession.exam_name.contains(exam_name))

            total = query.count()
            total_pages = max(1, (total + page_size - 1) // page_size)
            offset = (page - 1) * page_size

            alerts = query.order_by(AlertEvent.timestamp.desc()).offset(offset).limit(page_size).all()

            items = []
            for a in alerts:
                exam = a.exam_session
                student = exam.student if exam else None
                video_status = self._normalize_video_status(a)
                items.append({
                    "id": a.id,
                    "session_id": a.session_id,
                    "alert_type": a.alert_type,
                    "confidence": a.confidence,
                    "screenshot_path": a.screenshot_path,
                    "video_clip_path": a.video_clip_path,
                    "video_status": video_status,
                    "status": a.status,
                    "timestamp": a.timestamp.isoformat() if a.timestamp else None,
                    "description": a.description,
                    "exam_name": exam.exam_name if exam else None,
                    "student_id": exam.student_id if exam else None,
                    "student_name": student.name if student else None,
                    "student_no": student.student_no if student else None,
                })

            return {
                "items": items,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages
            }
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
                    alert_stats[atype] = {
                        "count": 0,
                        "avg_confidence": 0.0,
                        "confirmed": 0,
                        "false_positive": 0,
                        "pending": 0,
                        "effective": 0
                    }
                alert_stats[atype]["count"] += 1
                alert_stats[atype]["avg_confidence"] += alert.confidence
                if alert.status == AlertStatus.CONFIRMED_CHEATING.value:
                    alert_stats[atype]["confirmed"] += 1
                elif alert.status == AlertStatus.FALSE_POSITIVE.value:
                    alert_stats[atype]["false_positive"] += 1
                elif alert.status == AlertStatus.PENDING.value:
                    alert_stats[atype]["pending"] += 1

            for k in alert_stats:
                s = alert_stats[k]
                if s["count"] > 0:
                    s["avg_confidence"] = round(s["avg_confidence"] / s["count"], 4)
                s["effective"] = s["confirmed"] + s["pending"]

            timeline = []
            for alert in alerts:
                video_status = self._normalize_video_status(alert)
                timeline.append({
                    "id": alert.id,
                    "type": alert.alert_type,
                    "confidence": alert.confidence,
                    "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
                    "screenshot": alert.screenshot_path,
                    "video_clip": alert.video_clip_path,
                    "video_status": video_status,
                    "status": alert.status,
                    "description": alert.description
                })

            confirmed_count = sum(1 for a in alerts if a.status == AlertStatus.CONFIRMED_CHEATING.value)
            false_positive_count = sum(1 for a in alerts if a.status == AlertStatus.FALSE_POSITIVE.value)
            pending_count = sum(1 for a in alerts if a.status == AlertStatus.PENDING.value)

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
                "confirmed_cheating": confirmed_count,
                "false_positive": false_positive_count,
                "pending_review": pending_count,
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
        video_status = self._normalize_video_status(alert)
        alert_data = {
            "id": alert.id,
            "type": alert.alert_type,
            "confidence": alert.confidence,
            "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
            "screenshot": alert.screenshot_path,
            "video_clip": alert.video_clip_path,
            "video_status": video_status,
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

    def get_screenshot_local_path(self, url_path: str) -> Optional[str]:
        if not url_path:
            return None
        filename = url_path.replace("/alerts/", "")
        local = os.path.join(ALERTS_DIR, filename)
        return local if os.path.exists(local) else None


session_manager = SessionManager()
