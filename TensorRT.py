import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import serial  # 시리얼 통신 라이브러리
import time

# TensorRT 엔진 로드 클래스
class TRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(self.logger)

        # 엔진 로드
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()

        # 바인딩 이름 가져오기
        self.input_binding = self.engine.get_binding_name(0)  # 첫 번째 바인딩 이름
        self.output_binding = self.engine.get_binding_name(1)  # 두 번째 바인딩 이름

        # 입력/출력 버퍼 설정
        self.input_shape = self.engine.get_tensor_shape(self.input_binding)
        self.input_size = trt.volume(self.input_shape) * np.dtype(np.float32).itemsize
        self.output_shape = self.engine.get_tensor_shape(self.output_binding)
        self.output_size = trt.volume(self.output_shape) * np.dtype(np.float32).itemsize

        # GPU 메모리 할당
        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)
        self.stream = cuda.Stream()

    def infer(self, input_data):
        input_data = np.ascontiguousarray(input_data)
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()
        return output

# 이미지 전처리 함수
def preprocess_frame(frame, input_size=(640, 640)):
    original_frame = frame.copy()
    frame = cv2.resize(frame, input_size)
    frame = frame / 255.0
    frame = frame.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW
    input_data = np.expand_dims(frame, axis=0)  # 배치 차원 추가
    return np.ascontiguousarray(input_data), original_frame

# 추론 결과 후처리 함수
def postprocess(output, frame, input_size=(640, 640), conf_threshold=0.5, arduino=None):
    """
    추론 결과 후처리 및 바운딩 박스 그리기
    """
    h, w, _ = frame.shape
    detections = output[0]
    num_boxes = detections.shape[1]

    for i in range(num_boxes):
        x_center, y_center, width, height, confidence = detections[:, i]

        if confidence >= conf_threshold:
            x1 = int((x_center - width / 2) * w / input_size[0])
            y1 = int((y_center - height / 2) * h / input_size[1])
            x2 = int((x_center + width / 2) * w / input_size[0])
            y2 = int((y_center + height / 2) * h / input_size[1])

            # 중심점 계산
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 바운딩 박스와 중심점 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)  # 중심점 표시
            label = f"Conf: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 중심점 좌표를 아두이노로 전송
            if arduino:
                message = f"{center_x},{center_y}\n"
                arduino.write(message.encode())

    return frame

# 메인 함수
def main():
    engine_path = "best_fp16.trt"  # f16 최적화된 TensorRT 엔진
    input_size = (640, 640)
    conf_threshold = 0.5

    try:
        # 아두이노 시리얼 초기화
        arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=1)  # 포트와 보드레이트 설정
        time.sleep(2)  # 아두이노 초기화 대기

        # TensorRT 엔진 로드
        trt_infer = TRTInference(engine_path)

        # 웹캠 초기화
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break

            input_data, original_frame = preprocess_frame(frame, input_size)
            output = trt_infer.infer(input_data)
            result_frame = postprocess(output, original_frame, input_size, conf_threshold, arduino)

            # 결과 표시
            cv2.imshow("TensorRT Real-Time Detection", result_frame)

            # ESC 키로 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
