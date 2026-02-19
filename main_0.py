import cv2
import numpy as np
import time
import os

class MSESlideDetector:
    
    def __init__(self, video_path):
        """
        초기화: 영상 로드 및 프레임 추출
        
        Args:
            video_path: 분석할 영상 파일 경로
        """
        self.video_path = video_path
        self.load_time = 0  # 프레임 로드 소요 시간
        
        # ===== 프레임 로드 시간 측정 시작 =====
        load_start = time.time()
        
        # OpenCV로 영상 파일 열기
        self.cap = cv2.VideoCapture(video_path)
        
        # 영상 메타정보 추출
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # 초당 프레임 수
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 전체 프레임 수
        self.duration = self.total_frames / self.fps  # 영상 길이 (초)
        
        # 프레임 샘플링하여 메모리에 로드
        self.frames = self._load_frames()
        
        # ===== 프레임 로드 시간 측정 종료 =====
        self.load_time = time.time() - load_start
        
        print(f"[로드 완료]")
        print(f"  - 영상 길이: {self.duration:.1f}초")
        print(f"  - FPS: {self.fps:.2f}")
        print(f"  - 샘플링된 프레임: {len(self.frames)}개")
        print(f"  - 로드 시간: {self.load_time:.2f}초")
    
    def _load_frames(self, sample_rate=0.5):
        """
        영상에서 일정 간격으로 프레임 샘플링
        
        Args:
            sample_rate: 샘플링 간격 (초). 기본값 0.5초 = 1초에 2프레임
        
        Returns:
            list of tuples: [(프레임_인덱스, 타임스탬프, 프레임_이미지), ...]
        """
        frames = []
        
        # 샘플링 간격 계산 (프레임 단위)
        interval = int(self.fps * sample_rate)
        
        frame_idx = 0
        while self.cap.isOpened():
            # 프레임 읽기
            # ret: 성공 여부 (bool)
            # frame: 이미지 배열 (numpy array, BGR 형식)
            ret, frame = self.cap.read()
            
            if not ret:
                # 영상 끝 또는 읽기 실패
                break
            
            # 샘플링 간격에 해당하는 프레임만 저장
            if frame_idx % interval == 0:
                timestamp = frame_idx / self.fps  # 해당 프레임의 시간 (초)
                frames.append((frame_idx, timestamp, frame))
            
            frame_idx += 1
        
        # 영상 리소스 해제
        self.cap.release()
        
        return frames
    
    def detect(self, threshold=1000):
        """
        MSE 기반 슬라이드 변화 감지
        
        Args:
            threshold: 변화 판단 기준값
                - 기본값 1000 (경험적 최적값)
                - 낮출수록 민감 (과검출 위험)
                - 높일수록 둔감 (미검출 위험)
                - PPT 강의 권장 범위: 500 ~ 2000
        
        Returns:
            list of tuples: [(프레임_인덱스, 타임스탬프), ...]
        
        MSE 계산 공식:
            MSE = mean((prev_frame - curr_frame)^2)
        """
        changes = []  # 변화 감지된 프레임 목록
        prev_gray = None  # 이전 프레임 (그레이스케일)
        
        for frame_idx, timestamp, frame in self.frames:
            # ----- 전처리 -----
            # 1. BGR → 그레이스케일 변환 (연산량 감소)
            #    3채널 → 1채널로 줄여 3배 빠름
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 2. 리사이즈 (연산량 감소)
            #    원본 1920x1080 → 320x240으로 축소
            #    픽셀 수: 2,073,600 → 76,800 (약 27배 감소)
            gray = cv2.resize(gray, (320, 240))
            
            # ----- MSE 계산 및 변화 판단 -----
            if prev_gray is None:
                # 첫 프레임은 무조건 변화로 처리 (시작점)
                changes.append((frame_idx, timestamp))
            else:
                # MSE 계산
                # 1) 픽셀별 차이: prev - curr
                # 2) 제곱: 음수 제거 + 큰 차이 강조
                # 3) 평균: 전체 변화량을 단일 값으로
                diff = prev_gray.astype(float) - gray.astype(float)
                mse = np.mean(diff ** 2)
                
                # threshold 초과 시 슬라이드 전환으로 판단
                if mse > threshold:
                    changes.append((frame_idx, timestamp))
            
            # 현재 프레임을 다음 비교를 위해 저장
            prev_gray = gray
        
        return changes
    
    def _measure_save_time_per_image(self):
        """
        이미지 1개 저장 시간 측정 (서비스 시간 계산용)
        
        Returns:
            float: 이미지 1개 저장 평균 시간 (초)
        """
        if not self.frames:
            return 0
        
        # 테스트용 임시 폴더 생성
        test_dir = "temp_save_test"
        os.makedirs(test_dir, exist_ok=True)
        
        # 첫 번째 프레임으로 테스트
        frame = self.frames[0][2]
        
        # 5회 측정 후 평균 (정확도 향상)
        times = []
        for i in range(5):
            start = time.time()
            cv2.imwrite(f"{test_dir}/test_{i}.jpg", frame)
            times.append(time.time() - start)
        
        # 임시 폴더 삭제
        import shutil
        shutil.rmtree(test_dir)
        
        return sum(times) / len(times)
    
    def save_slides(self, changes, output_dir='slides'):
        """
        감지된 슬라이드 프레임을 이미지로 저장
        
        Args:
            changes: detect()의 반환값 [(프레임_인덱스, 타임스탬프), ...]
            output_dir: 저장 폴더 경로
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 프레임 인덱스 → 이미지 매핑 딕셔너리
        frame_dict = {f[0]: f[2] for f in self.frames}
        
        saved_count = 0
        for i, (frame_idx, timestamp) in enumerate(changes):
            if frame_idx in frame_dict:
                frame = frame_dict[frame_idx]
                
                # 파일명: slide_순번_타임스탬프.jpg
                filename = f"{output_dir}/slide_{i:03d}_{timestamp:.1f}s.jpg"
                
                # JPEG 형식으로 저장
                cv2.imwrite(filename, frame)
                saved_count += 1
        
        return saved_count
    
    def run(self, threshold=1000, output_dir='slides'):
        """
        전체 파이프라인 실행 (감지 + 저장 + 리포트)
        
        Args:
            threshold: MSE 변화 감지 기준값
            output_dir: 결과 저장 폴더
        
        Returns:
            dict: 실행 결과 및 시간 측정 정보
        """
        print(f"\n[MSE 슬라이드 감지 시작]")
        print(f"  - Threshold: {threshold}")
        
        # ===== 1. 저장 시간 측정 (이미지 1개당) =====
        save_time_per_image = self._measure_save_time_per_image()
        
        # ===== 2. 슬라이드 변화 감지 (알고리즘 연산) =====
        algo_start = time.time()
        changes = self.detect(threshold=threshold)
        algo_time = time.time() - algo_start
        
        print(f"\n[감지 완료]")
        print(f"  - 감지된 슬라이드: {len(changes)}개")
        print(f"  - 알고리즘 연산 시간: {algo_time:.3f}초")
        
        # ===== 3. 슬라이드 이미지 저장 =====
        save_start = time.time()
        saved_count = self.save_slides(changes, output_dir)
        actual_save_time = time.time() - save_start
        
        print(f"\n[저장 완료]")
        print(f"  - 저장된 이미지: {saved_count}개")
        print(f"  - 저장 시간: {actual_save_time:.3f}초")
        print(f"  - 저장 위치: {output_dir}/")
        
        # ===== 4. 총 서비스 시간 계산 =====
        total_service_time = self.load_time + algo_time + actual_save_time
        
        print(f"\n[시간 분석]")
        print(f"  - 프레임 로드: {self.load_time:.2f}초")
        print(f"  - 알고리즘 연산: {algo_time:.2f}초")
        print(f"  - 이미지 저장: {actual_save_time:.2f}초")
        print(f"  --------------------------------")
        print(f"  - 총 서비스 시간: {total_service_time:.2f}초")
        
        # ===== 5. 리포트 생성 =====
        self._generate_report(changes, output_dir, {
            'threshold': threshold,
            'load_time': self.load_time,
            'algo_time': algo_time,
            'save_time': actual_save_time,
            'total_service_time': total_service_time
        })
        
        return {
            'changes': changes,
            'count': len(changes),
            'timestamps': [t for _, t in changes],
            'load_time': self.load_time,
            'algo_time': algo_time,
            'save_time': actual_save_time,
            'total_service_time': total_service_time
        }
    
    def _generate_report(self, changes, output_dir, time_info):
        """
        결과 리포트 생성
        
        Args:
            changes: 감지된 슬라이드 목록
            output_dir: 저장 폴더
            time_info: 시간 측정 정보 딕셔너리
        """
        report_path = f"{output_dir}/report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("MSE 슬라이드 변화 감지 리포트\n")
            f.write("=" * 50 + "\n\n")
            
            # 영상 정보
            f.write("[영상 정보]\n")
            f.write(f"  파일: {self.video_path}\n")
            f.write(f"  길이: {self.duration:.1f}초\n")
            f.write(f"  FPS: {self.fps:.2f}\n")
            f.write(f"  샘플링된 프레임: {len(self.frames)}개\n\n")
            
            # 감지 설정
            f.write("[감지 설정]\n")
            f.write(f"  알고리즘: MSE (Mean Squared Error)\n")
            f.write(f"  Threshold: {time_info['threshold']}\n")
            f.write(f"  샘플링 간격: 0.5초\n\n")
            
            # 감지 결과
            f.write("[감지 결과]\n")
            f.write(f"  감지된 슬라이드: {len(changes)}개\n\n")
            
            # 시간 분석
            f.write("[시간 분석]\n")
            f.write(f"  프레임 로드: {time_info['load_time']:.2f}초\n")
            f.write(f"  알고리즘 연산: {time_info['algo_time']:.2f}초\n")
            f.write(f"  이미지 저장: {time_info['save_time']:.2f}초\n")
            f.write(f"  총 서비스 시간: {time_info['total_service_time']:.2f}초\n\n")
            
            # 감지된 타임스탬프
            f.write("[감지된 타임스탬프]\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'번호':>4} {'타임스탬프':>12} {'시:분:초':>12}\n")
            f.write("-" * 30 + "\n")
            
            for i, (frame_idx, timestamp) in enumerate(changes):
                # 초 → 시:분:초 변환
                minutes, seconds = divmod(timestamp, 60)
                hours, minutes = divmod(minutes, 60)
                time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
                
                f.write(f"{i+1:>4} {timestamp:>11.1f}초 {time_str:>12}\n")
        
        print(f"\n[리포트 저장] {report_path}")


# ===== 실행 =====
if __name__ == "__main__":
    # 영상 파일 경로
    VIDEO_PATH = "lecture.mp4"
    
    # 설정값
    THRESHOLD = 500  # MSE 변화 감지 기준 (500~2000 권장)
    OUTPUT_DIR = "output"  # 결과 저장 폴더
    
    # 감지기 생성 및 실행
    detector = MSESlideDetector(VIDEO_PATH)
    result = detector.run(threshold=THRESHOLD, output_dir=OUTPUT_DIR)
    
    # 결과 요약
    print(f"\n{'='*50}")
    print(f"완료! {result['count']}개 슬라이드 추출됨")
    print(f"저장 위치: {OUTPUT_DIR}/")
    print(f"{'='*50}")