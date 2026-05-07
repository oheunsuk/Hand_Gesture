"""OpenCV 웹캠 열기: Windows에서 인덱스·백엔드 차이를 줄인다."""
from __future__ import annotations

import sys

import cv2


def open_webcam(max_index: int = 4) -> cv2.VideoCapture:
    """
    VideoCapture(0)만으로 실패하는 환경이 있어,
    Windows에서는 CAP_DSHOW를 우선 시도하고 인덱스를 순회한다.
    """
    apis: list[int] = []
    if sys.platform == "win32":
        apis.append(cv2.CAP_DSHOW)
    apis.append(cv2.CAP_ANY)

    for api in apis:
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, api)
            if cap.isOpened():
                return cap
            cap.release()

    raise RuntimeError(
        "웹캠을 열 수 없습니다. USB/내장 카메라 연결, 다른 앱의 카메라 사용 종료, "
        "Windows 설정 > 개인 정보 보호 > 카메라에서 데스크톱 앱 허용을 확인하세요."
    )
