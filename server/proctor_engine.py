import time
import math
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp

from .session_manager import session_manager, AlertType


@dataclass
class DetectionResult:
    has_face: bool
    face_count: int
    face_center: Optional[Tuple[float, float]]
    gaze_direction: Optional[Tuple[float, float]]
    is_off_center: bool
    is_gaze_deviated: bool
    face_encoding: Optional[np.ndarray]
    confidence: float
    landmarks: Optional[Any] = None


class ProctorEngine:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.OFF_CENTER_THRESHOLD = 0.25
        self.GAZE_ANGLE_THRESHOLD = 30.0
        self.NO_FACE_DURATION_THRESHOLD = 5.0
        self.GAZE_DEVIATION_DURATION_THRESHOLD = 3.0
        self.FACE_SIMILARITY_THRESHOLD = 0.6
        self.LOW_CONFIDENCE_THRESHOLD = 0.7

        self.IRIS_LEFT = [468, 469, 470, 471, 472]
        self.IRIS_RIGHT = [473, 474, 475, 476, 477]
        self.EYE_LEFT_TOP = 386
        self.EYE_LEFT_BOTTOM = 374
        self.EYE_LEFT_LEFT = 362
        self.EYE_LEFT_RIGHT = 263
        self.EYE_RIGHT_TOP = 159
        self.EYE_RIGHT_BOTTOM = 145
        self.EYE_RIGHT_LEFT = 33
        self.EYE_RIGHT_RIGHT = 133
        self.NOSE_TIP = 1
        self.FOREHEAD = 10
        self.CHIN = 152

    def detect(self, frame: np.ndarray) -> DetectionResult:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        detection_result = self.face_mesh.process(rgb_frame)
        face_mesh_results = detection_result.multi_face_landmarks

        face_count = len(face_mesh_results) if face_mesh_results else 0
        has_face = face_count > 0

        if not has_face:
            return DetectionResult(
                has_face=False,
                face_count=0,
                face_center=None,
                gaze_direction=None,
                is_off_center=False,
                is_gaze_deviated=False,
                face_encoding=None,
                confidence=0.0
            )

        primary_landmarks = face_mesh_results[0]

        nose_tip = primary_landmarks.landmark[self.NOSE_TIP]
        face_center_x = nose_tip.x
        face_center_y = nose_tip.y

        face_center = (face_center_x, face_center_y)

        center_deviation = math.sqrt(
            (face_center_x - 0.5) ** 2 + (face_center_y - 0.5) ** 2
        )
        is_off_center = center_deviation > self.OFF_CENTER_THRESHOLD

        gaze_yaw, gaze_pitch = self._estimate_gaze(primary_landmarks, w, h)
        gaze_direction = (gaze_yaw, gaze_pitch)
        gaze_magnitude = math.sqrt(gaze_yaw ** 2 + gaze_pitch ** 2)
        is_gaze_deviated = gaze_magnitude > self.GAZE_ANGLE_THRESHOLD

        face_encoding = self._extract_face_encoding(primary_landmarks)

        confidence = self._calculate_confidence(
            face_count, center_deviation, gaze_magnitude, primary_landmarks
        )

        return DetectionResult(
            has_face=True,
            face_count=face_count,
            face_center=face_center,
            gaze_direction=gaze_direction,
            is_off_center=is_off_center,
            is_gaze_deviated=is_gaze_deviated,
            face_encoding=face_encoding,
            confidence=confidence,
            landmarks=primary_landmarks
        )

    def _estimate_gaze(self, landmarks, img_w: int, img_h: int) -> Tuple[float, float]:
        iris_left = landmarks.landmark[self.IRIS_LEFT[0]]
        iris_right = landmarks.landmark[self.IRIS_RIGHT[0]]

        eye_left_left = landmarks.landmark[self.EYE_LEFT_LEFT]
        eye_left_right = landmarks.landmark[self.EYE_LEFT_RIGHT]
        eye_right_left = landmarks.landmark[self.EYE_RIGHT_LEFT]
        eye_right_right = landmarks.landmark[self.EYE_RIGHT_RIGHT]

        eye_left_center_x = (eye_left_left.x + eye_left_right.x) / 2
        eye_left_center_y = (eye_left_left.y + eye_left_right.y) / 2
        eye_right_center_x = (eye_right_left.x + eye_right_right.x) / 2
        eye_right_center_y = (eye_right_left.y + eye_right_right.y) / 2

        left_offset_x = iris_left.x - eye_left_center_x
        left_offset_y = iris_left.y - eye_left_center_y
        right_offset_x = iris_right.x - eye_right_center_x
        right_offset_y = iris_right.y - eye_right_center_y

        avg_offset_x = (left_offset_x + right_offset_x) / 2
        avg_offset_y = (left_offset_y + right_offset_y) / 2

        eye_width_left = abs(eye_left_right.x - eye_left_left.x)
        eye_width_right = abs(eye_right_right.x - eye_right_left.x)
        avg_eye_width = (eye_width_left + eye_width_right) / 2

        eye_height_left = abs(landmarks.landmark[self.EYE_LEFT_TOP].y - landmarks.landmark[self.EYE_LEFT_BOTTOM].y)
        eye_height_right = abs(landmarks.landmark[self.EYE_RIGHT_TOP].y - landmarks.landmark[self.EYE_RIGHT_BOTTOM].y)
        avg_eye_height = (eye_height_left + eye_height_right) / 2

        yaw_ratio = avg_offset_x / avg_eye_width if avg_eye_width > 0 else 0
        pitch_ratio = avg_offset_y / avg_eye_height if avg_eye_height > 0 else 0

        yaw_angle = yaw_ratio * 60.0
        pitch_angle = pitch_ratio * 40.0

        return yaw_angle, pitch_angle

    def _extract_face_encoding(self, landmarks) -> np.ndarray:
        key_points = [
            self.NOSE_TIP, self.FOREHEAD, self.CHIN,
            self.EYE_LEFT_LEFT, self.EYE_LEFT_RIGHT,
            self.EYE_RIGHT_LEFT, self.EYE_RIGHT_RIGHT,
            234, 454, 10, 152
        ]

        encoding = []
        for idx in key_points:
            pt = landmarks.landmark[idx]
            encoding.extend([pt.x, pt.y, pt.z])

        encoding = np.array(encoding, dtype=np.float64)
        encoding = encoding / (np.linalg.norm(encoding) + 1e-8)
        return encoding

    def _calculate_confidence(self, face_count: int, center_deviation: float,
                               gaze_magnitude: float, landmarks) -> float:
        confidence = 1.0

        if face_count > 1:
            confidence = min(confidence, 0.95)

        center_factor = max(0.0, 1.0 - (center_deviation / 0.5))
        confidence *= center_factor

        gaze_factor = max(0.0, 1.0 - (gaze_magnitude / 60.0))
        confidence *= gaze_factor

        confidence = max(0.1, min(1.0, confidence))
        return round(confidence, 4)

    def verify_face(self, frame: np.ndarray, stored_encoding: np.ndarray) -> Tuple[bool, float]:
        result = self.detect(frame)
        if not result.has_face or result.face_encoding is None:
            return False, 0.0

        similarity = self._cosine_similarity(result.face_encoding, stored_encoding)
        is_match = similarity >= self.FACE_SIMILARITY_THRESHOLD
        return is_match, round(similarity, 4)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if vec1.shape != vec2.shape:
            return 0.0
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def process_frame(self, session_id: str, frame: np.ndarray) -> List[Dict[str, Any]]:
        state = session_manager.get_session_state(session_id)
        if not state:
            return []

        session_manager.push_frame_to_buffer(session_id, frame)

        result = self.detect(frame)
        alerts = []
        now = time.time()

        if not result.has_face:
            if state.no_face_start_time == 0:
                state.no_face_start_time = now
            else:
                duration = now - state.no_face_start_time
                if duration >= self.NO_FACE_DURATION_THRESHOLD:
                    alert_confidence = min(0.95, 0.5 + duration * 0.05)
                    screenshot = session_manager.save_screenshot(
                        session_id, frame, AlertType.NO_FACE.value
                    )
                    alert = session_manager.add_alert(
                        session_id=session_id,
                        alert_type=AlertType.NO_FACE,
                        confidence=alert_confidence,
                        screenshot_path=screenshot,
                        description=f"画面中无人脸持续 {duration:.1f} 秒"
                    )
                    alerts.append({
                        "id": alert.id,
                        "type": AlertType.NO_FACE.value,
                        "confidence": alert_confidence,
                        "screenshot": screenshot,
                        "video_clip": alert.video_clip_path,
                        "needs_review": alert_confidence < self.LOW_CONFIDENCE_THRESHOLD
                    })
                    state.no_face_start_time = 0
        else:
            state.no_face_start_time = 0

        if result.face_count > 1:
            screenshot = session_manager.save_screenshot(
                session_id, frame, AlertType.MULTI_FACE.value
            )
            alert = session_manager.add_alert(
                session_id=session_id,
                alert_type=AlertType.MULTI_FACE,
                confidence=0.98,
                screenshot_path=screenshot,
                description=f"检测到 {result.face_count} 张人脸"
            )
            alerts.append({
                "id": alert.id,
                "type": AlertType.MULTI_FACE.value,
                "confidence": 0.98,
                "screenshot": screenshot,
                "video_clip": alert.video_clip_path,
                "needs_review": False
            })

        if result.has_face and result.is_off_center:
            screenshot = session_manager.save_screenshot(
                session_id, frame, AlertType.FACE_OFF_CENTER.value
            )
            alert = session_manager.add_alert(
                session_id=session_id,
                alert_type=AlertType.FACE_OFF_CENTER,
                confidence=0.85,
                screenshot_path=screenshot,
                description="人脸偏离画面中心"
            )
            alerts.append({
                "id": alert.id,
                "type": AlertType.FACE_OFF_CENTER.value,
                "confidence": 0.85,
                "screenshot": screenshot,
                "video_clip": alert.video_clip_path,
                "needs_review": True
            })

        if result.has_face and result.is_gaze_deviated:
            if state.gaze_deviation_start_time == 0:
                state.gaze_deviation_start_time = now
            else:
                duration = now - state.gaze_deviation_start_time
                if duration >= self.GAZE_DEVIATION_DURATION_THRESHOLD:
                    gaze_yaw, gaze_pitch = result.gaze_direction
                    gaze_magnitude = math.sqrt(gaze_yaw ** 2 + gaze_pitch ** 2)
                    alert_confidence = min(0.95, 0.6 + (gaze_magnitude - 30) * 0.01)
                    screenshot = session_manager.save_screenshot(
                        session_id, frame, AlertType.GAZE_DEVIATION.value
                    )
                    alert = session_manager.add_alert(
                        session_id=session_id,
                        alert_type=AlertType.GAZE_DEVIATION,
                        confidence=alert_confidence,
                        screenshot_path=screenshot,
                        description=f"视线偏离持续 {duration:.1f} 秒，角度 {gaze_magnitude:.1f}°"
                    )
                    alerts.append({
                        "id": alert.id,
                        "type": AlertType.GAZE_DEVIATION.value,
                        "confidence": alert_confidence,
                        "screenshot": screenshot,
                        "video_clip": alert.video_clip_path,
                        "needs_review": alert_confidence < self.LOW_CONFIDENCE_THRESHOLD
                    })
                    state.gaze_deviation_start_time = 0
        else:
            state.gaze_deviation_start_time = 0

        if result.has_face and result.face_encoding is not None:
            stored_encoding = session_manager.get_student_face(state.student_id)
            if stored_encoding is not None and state.last_face_encoding is not None:
                similarity = self._cosine_similarity(result.face_encoding, state.last_face_encoding)
                if similarity < self.FACE_SIMILARITY_THRESHOLD * 0.8:
                    stored_similarity = self._cosine_similarity(result.face_encoding, stored_encoding)
                    if stored_similarity < self.FACE_SIMILARITY_THRESHOLD:
                        alert_confidence = max(0.7, 1.0 - stored_similarity)
                        screenshot = session_manager.save_screenshot(
                            session_id, frame, AlertType.FACE_REPLACEMENT.value
                        )
                        alert = session_manager.add_alert(
                            session_id=session_id,
                            alert_type=AlertType.FACE_REPLACEMENT,
                            confidence=alert_confidence,
                            screenshot_path=screenshot,
                            description=f"疑似人员更换，相似度 {stored_similarity:.3f}"
                        )
                        alerts.append({
                            "id": alert.id,
                            "type": AlertType.FACE_REPLACEMENT.value,
                            "confidence": alert_confidence,
                            "screenshot": screenshot,
                            "video_clip": alert.video_clip_path,
                            "needs_review": alert_confidence < self.LOW_CONFIDENCE_THRESHOLD
                        })

            if state.last_face_encoding is None:
                state.last_face_encoding = result.face_encoding.copy()

        state.last_frame_time = now
        return alerts

    def draw_overlay(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        if not result.has_face or not result.landmarks:
            return frame

        h, w = frame.shape[:2]
        overlay = frame.copy()

        self.mp_drawing.draw_landmarks(
            image=overlay,
            landmark_list=result.landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=1, circle_radius=1
            )
        )

        if result.face_center:
            cx = int(result.face_center[0] * w)
            cy = int(result.face_center[1] * h)
            color = (0, 0, 255) if result.is_off_center else (0, 255, 0)
            cv2.circle(overlay, (cx, cy), 5, color, -1)

        if result.is_gaze_deviated:
            cv2.putText(
                overlay, "GAZE DEVIATED", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        if result.face_count > 1:
            cv2.putText(
                overlay, f"FACES: {result.face_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        cv2.putText(
            overlay, f"Conf: {result.confidence:.2f}", (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

        return overlay


proctor_engine = ProctorEngine()
