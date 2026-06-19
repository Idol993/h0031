import os
import io
import json
import time
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .session_manager import (
    session_manager,
    AlertType,
    AlertStatus,
    VideoStatus,
    ExamStatus,
    ALERTS_DIR,
    BASE_DIR
)
from .proctor_engine import proctor_engine


app = FastAPI(title="在线考试监考系统 API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(ALERTS_DIR, exist_ok=True)
app.mount("/alerts", StaticFiles(directory=ALERTS_DIR), name="alerts")


# ---------- Pydantic Models ----------

class StudentCreate(BaseModel):
    name: str
    student_no: str


class StudentResponse(BaseModel):
    id: str
    name: str
    student_no: str
    has_face_encoding: bool


class ExamCreate(BaseModel):
    student_id: str
    exam_name: str


class ExamResponse(BaseModel):
    id: str
    student_id: str
    exam_name: str
    status: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    verify_confidence: float
    total_alerts: int


class AlertResponse(BaseModel):
    id: str
    session_id: str
    alert_type: str
    confidence: float
    screenshot_path: Optional[str] = None
    video_clip_path: Optional[str] = None
    video_status: str
    status: str
    timestamp: Optional[str] = None
    description: Optional[str] = None


class AlertStatusUpdate(BaseModel):
    status: AlertStatus


class VerifyResponse(BaseModel):
    verified: bool
    confidence: float
    session_id: Optional[str] = None


class ReportResponse(BaseModel):
    session_id: str
    student_id: str
    exam_name: str
    status: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: int
    verify_confidence: float
    total_alerts: int
    confirmed_cheating: int
    false_positive: int
    pending_review: int
    alert_stats: Dict[str, Any]
    overall_confidence_score: float
    timeline: List[Dict[str, Any]]


class PaginatedAlertResponse(BaseModel):
    items: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    total_pages: int


# ---------- Utility Functions ----------

def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


# ---------- Student Routes ----------

@app.post("/api/students", response_model=StudentResponse)
async def create_student(student: StudentCreate):
    db_student = session_manager.create_student(student.name, student.student_no)
    return StudentResponse(
        id=db_student.id,
        name=db_student.name,
        student_no=db_student.student_no,
        has_face_encoding=db_student.face_encoding is not None
    )


@app.post("/api/students/{student_id}/face")
async def upload_student_face(student_id: str, file: UploadFile = File(...)):
    contents = await file.read()
    img = decode_image(contents)
    if img is None:
        raise HTTPException(status_code=400, detail="无效的图片文件")

    result = proctor_engine.detect(img)
    if not result.has_face or result.face_encoding is None:
        raise HTTPException(status_code=400, detail="未检测到人脸")

    success = session_manager.set_student_face(student_id, result.face_encoding)
    if not success:
        raise HTTPException(status_code=404, detail="考生不存在")

    return {"success": True, "face_detected": True, "confidence": result.confidence}


@app.get("/api/students/{student_id}", response_model=StudentResponse)
async def get_student(student_id: str):
    from .session_manager import SessionLocal, Student
    db = SessionLocal()
    try:
        student = db.query(Student).filter(Student.id == student_id).first()
        if not student:
            raise HTTPException(status_code=404, detail="考生不存在")
        return StudentResponse(
            id=student.id,
            name=student.name,
            student_no=student.student_no,
            has_face_encoding=student.face_encoding is not None
        )
    finally:
        db.close()


# ---------- Exam Session Routes ----------

@app.post("/api/exams", response_model=ExamResponse)
async def create_exam(exam: ExamCreate):
    session = session_manager.create_exam_session(exam.student_id, exam.exam_name)
    return ExamResponse(
        id=session.id,
        student_id=session.student_id,
        exam_name=session.exam_name,
        status=session.status,
        verify_confidence=session.verify_confidence,
        total_alerts=session.total_alerts
    )


@app.post("/api/exams/{session_id}/verify")
async def verify_face(session_id: str, file: UploadFile = File(...)):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="考试会话不存在")

    if session.status == ExamStatus.IN_PROGRESS.value:
        raise HTTPException(status_code=400, detail="考试已在进行中")

    contents = await file.read()
    img = decode_image(contents)
    if img is None:
        raise HTTPException(status_code=400, detail="无效的图片文件")

    stored_encoding = session_manager.get_student_face(session.student_id)
    if stored_encoding is None:
        raise HTTPException(status_code=400, detail="考生未录入人脸信息")

    is_match, confidence = proctor_engine.verify_face(img, stored_encoding)
    passed = confidence >= 0.9

    if passed:
        session_manager.start_exam(session_id, verify_confidence=confidence)

    return VerifyResponse(
        verified=passed,
        confidence=confidence,
        session_id=session_id if passed else None
    )


@app.post("/api/exams/{session_id}/start")
async def start_exam(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="考试会话不存在")

    success = session_manager.start_exam(session_id, verify_confidence=0.0)
    if not success:
        raise HTTPException(status_code=400, detail="无法开始考试")

    return {"success": True, "session_id": session_id}


@app.post("/api/exams/{session_id}/end")
async def end_exam(session_id: str):
    success = session_manager.end_exam(session_id, terminated=False)
    if not success:
        raise HTTPException(status_code=404, detail="考试会话不存在")
    return {"success": True, "session_id": session_id}


@app.get("/api/exams/{session_id}", response_model=ExamResponse)
async def get_exam(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="考试会话不存在")
    return ExamResponse(
        id=session.id,
        student_id=session.student_id,
        exam_name=session.exam_name,
        status=session.status,
        start_time=session.start_time.isoformat() if session.start_time else None,
        end_time=session.end_time.isoformat() if session.end_time else None,
        verify_confidence=session.verify_confidence,
        total_alerts=session.total_alerts
    )


@app.get("/api/exams", response_model=List[ExamResponse])
async def list_exams():
    sessions = session_manager.get_all_sessions()
    return [
        ExamResponse(
            id=s.id,
            student_id=s.student_id,
            exam_name=s.exam_name,
            status=s.status,
            start_time=s.start_time.isoformat() if s.start_time else None,
            end_time=s.end_time.isoformat() if s.end_time else None,
            verify_confidence=s.verify_confidence,
            total_alerts=s.total_alerts
        )
        for s in sessions
    ]


# ---------- Alert Routes ----------

@app.get("/api/exams/{session_id}/alerts", response_model=List[AlertResponse])
async def get_alerts(session_id: str, status: Optional[AlertStatus] = None):
    alerts = session_manager.get_alerts(session_id, status)
    return [
        AlertResponse(
            id=a.id,
            session_id=a.session_id,
            alert_type=a.alert_type,
            confidence=a.confidence,
            screenshot_path=a.screenshot_path,
            video_clip_path=a.video_clip_path,
            video_status=a.video_status,
            status=a.status,
            timestamp=a.timestamp.isoformat() if a.timestamp else None,
            description=a.description
        )
        for a in alerts
    ]


@app.put("/api/alerts/{alert_id}/status")
async def update_alert_status(alert_id: str, update: AlertStatusUpdate):
    success = session_manager.update_alert_status(alert_id, update.status)
    if not success:
        raise HTTPException(status_code=404, detail="告警不存在")

    result = {"success": True, "alert_id": alert_id, "status": update.status.value}

    if update.status == AlertStatus.CONFIRMED_CHEATING:
        result["message"] = "已确认作弊，考试已终止"

    return result


# ---------- Aggregated Alert Query ----------

@app.get("/api/alerts", response_model=PaginatedAlertResponse)
async def query_alerts(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    session_id: Optional[str] = Query(None, description="考试会话ID"),
    student_id: Optional[str] = Query(None, description="学生ID"),
    alert_type: Optional[AlertType] = Query(None, description="异常类型"),
    status: Optional[AlertStatus] = Query(None, description="处理状态"),
    exam_name: Optional[str] = Query(None, description="考试名称（模糊搜索）")
):
    result = session_manager.query_alerts_paginated(
        page=page,
        page_size=page_size,
        session_id=session_id,
        student_id=student_id,
        alert_type=alert_type,
        status=status,
        exam_name=exam_name
    )
    return PaginatedAlertResponse(**result)


# ---------- Report Routes ----------

@app.get("/api/exams/{session_id}/report")
async def get_report(session_id: str):
    report = session_manager.generate_report(session_id)
    if not report:
        raise HTTPException(status_code=404, detail="考试会话不存在")
    return report


@app.get("/api/exams/{session_id}/report.pdf")
async def get_report_pdf(session_id: str):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch, cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image,
            Table, TableStyle, PageBreak, HRFlowable
        )
        from reportlab.lib import colors
        from reportlab.lib.utils import ImageReader
    except ImportError:
        raise HTTPException(status_code=500, detail="PDF生成库未安装")

    report = session_manager.generate_report(session_id)
    if not report:
        raise HTTPException(status_code=404, detail="考试会话不存在")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=20
    )

    story.append(Paragraph("在线考试监考报告", title_style))
    story.append(Spacer(1, 0.2 * inch))

    info_data = [
        ["考试名称", report.get("exam_name", "")],
        ["会话ID", session_id],
        ["学生ID", report.get("student_id", "")],
        ["考试状态", report.get("status", "")],
        ["开始时间", report.get("start_time", "-")],
        ["结束时间", report.get("end_time", "-")],
        ["考试时长(秒)", str(report.get("duration_seconds", 0))],
        ["人脸核验置信度", f"{report.get('verify_confidence', 0):.2%}"],
        ["告警总数", str(report.get("total_alerts", 0))],
        ["确认作弊", str(report.get("confirmed_cheating", 0))],
        ["误报", str(report.get("false_positive", 0))],
        ["待复核", str(report.get("pending_review", 0))],
        ["整体置信度评分", f"{report.get('overall_confidence_score', 0):.2%}"],
    ]

    info_table = Table(info_data, colWidths=[2 * inch, 4 * inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("告警统计", styles['Heading2']))
    story.append(Spacer(1, 0.1 * inch))

    alert_stats = report.get("alert_stats", {})
    if alert_stats:
        stats_data = [["告警类型", "次数", "平均置信度", "确认作弊", "误报", "待复核", "有效异常"]]
        for atype, stats in alert_stats.items():
            stats_data.append([
                atype,
                str(stats.get("count", 0)),
                f"{stats.get('avg_confidence', 0):.2%}",
                str(stats.get("confirmed", 0)),
                str(stats.get("false_positive", 0)),
                str(stats.get("pending", 0)),
                str(stats.get("effective", 0)),
            ])

        stats_table = Table(stats_data, colWidths=[1.2 * inch, 0.6 * inch, 1 * inch, 0.8 * inch, 0.6 * inch, 0.7 * inch, 0.8 * inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        story.append(stats_table)
    else:
        story.append(Paragraph("无告警记录", styles['Normal']))

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("告警时间线（含留证材料）", styles['Heading2']))
    story.append(Spacer(1, 0.1 * inch))

    timeline = report.get("timeline", [])
    for i, event in enumerate(timeline):
        status_color = "#FF0000" if event.get("status") == "confirmed_cheating" else (
            "#999999" if event.get("status") == "false_positive" else "#000000"
        )

        event_text = (
            f'<font color="{status_color}">'
            f"[{event.get('timestamp', '')}] {event.get('type', '')} - "
            f"置信度: {event.get('confidence', 0):.2%} - "
            f"状态: {event.get('status', '')}"
            f"</font><br/>"
            f"描述: {event.get('description', '')}"
        )
        story.append(Paragraph(event_text, styles['Normal']))

        screenshot_url = event.get("screenshot")
        if screenshot_url:
            local_path = session_manager.get_screenshot_local_path(screenshot_url)
            if local_path and os.path.exists(local_path):
                try:
                    img = Image(local_path, width=3 * inch, height=2.25 * inch)
                    img.hAlign = 'LEFT'
                    story.append(img)
                except Exception:
                    story.append(Paragraph(f"截图: {screenshot_url}", styles['Normal']))
            else:
                story.append(Paragraph(f"截图: {screenshot_url}", styles['Normal']))

        video_clip_url = event.get("video_clip")
        video_status = event.get("video_status", VideoStatus.NONE.value)
        status_label_map = {
            VideoStatus.READY.value: "✅ 视频片段 (已生成)",
            VideoStatus.GENERATING.value: "⏳ 视频片段 (生成中，稍后查看)",
            VideoStatus.FAILED.value: "❌ 视频片段 (生成失败)",
            VideoStatus.NONE.value: "⚠ 无视频片段",
        }
        video_label = status_label_map.get(video_status, f"视频状态: {video_status}")

        if video_clip_url:
            story.append(Paragraph(f"{video_label}: {video_clip_url}", styles['Normal']))
        else:
            story.append(Paragraph(video_label, styles['Normal']))

        story.append(Spacer(1, 0.15 * inch))

        if i > 0 and i % 5 == 0:
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))

    doc.build(story)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_{session_id}.pdf"}
    )


# ---------- WebSocket Video Stream ----------

@app.websocket("/ws/proctor/{session_id}")
async def websocket_proctor(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="考试会话不存在")
        return

    if session.status != ExamStatus.IN_PROGRESS.value:
        await websocket.close(code=4003, reason="考试未开始或已结束")
        return

    state = session_manager.get_session_state(session_id)
    if not state:
        session_manager.start_exam(session_id, session.verify_confidence)
        state = session_manager.get_session_state(session_id)

    frame_count = 0
    last_process_time = 0
    process_interval = 1.0

    try:
        while True:
            data = await websocket.receive_bytes()

            if not data:
                continue

            if not session_manager.is_exam_active(session_id):
                current = session_manager.get_session(session_id)
                status_msg = current.status if current else "unknown"
                await websocket.send_json({
                    "type": "exam_terminated",
                    "message": f"考试已结束（状态: {status_msg}），拒绝接收视频帧",
                    "status": status_msg
                })
                await websocket.close(code=4003, reason=f"考试已终止: {status_msg}")
                return

            img = decode_image(data)
            if img is None:
                await websocket.send_json({"type": "error", "message": "无效的帧数据"})
                continue

            now = time.time()
            should_process = (now - last_process_time) >= process_interval

            alerts = []
            if should_process:
                last_process_time = now
                frame_count += 1
                alerts = proctor_engine.process_frame(session_id, img)

            result = proctor_engine.detect(img)

            response = {
                "type": "frame_result",
                "frame_id": frame_count,
                "has_face": result.has_face,
                "face_count": result.face_count,
                "confidence": result.confidence,
                "is_off_center": result.is_off_center,
                "is_gaze_deviated": result.is_gaze_deviated,
                "alerts": alerts,
                "timestamp": datetime.utcnow().isoformat()
            }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.close(code=4000, reason=str(e))
        except Exception:
            pass
    finally:
        try:
            session_manager.flush_all_pending_recordings(session_id)
        except Exception:
            pass


# ---------- SSE Alert Stream ----------

@app.get("/sse/alerts/{session_id}")
async def sse_alerts(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="考试会话不存在")

    async def event_generator():
        queue = asyncio.Queue()

        def callback(alert_data):
            asyncio.run_coroutine_threadsafe(queue.put(alert_data), asyncio.get_event_loop())

        session_manager.register_sse_callback(session_id, callback)

        try:
            yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n"

            while True:
                alert = await queue.get()
                yield f"data: {json.dumps({'type': 'alert', 'data': alert})}\n\n"
        except asyncio.CancelledError:
            session_manager.unregister_sse_callback(session_id, callback)
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


# ---------- Health Check ----------

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "active_sessions": len(session_manager._active_sessions)
    }


# ---------- Main Entry ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
