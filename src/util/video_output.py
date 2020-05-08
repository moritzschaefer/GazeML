import queue
import threading
import time

import cv2 as cv


class RecordVideoThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, daemon=None):
        super(RecordVideoThread, self).__init__(group, target=self._record_frame,
                                                name=name, args=args, kwargs=kwargs, daemon=daemon)
        self.video_out_done = threading.Condition()
        self.video_out = None
        self.video_out_should_stop = False
        self.video_out_queue = queue.Queue()

    def _record_frame(self, output_path, data_source):
        last_frame_time = None
        out_fps = 30
        out_frame_interval = 1.0 / out_fps
        while not self.video_out_should_stop:
            frame_index = self.video_out_queue.get()
            if frame_index is None:
                break
            assert frame_index in data_source._frames
            frame = data_source._frames[frame_index]['bgr']
            h, w, _ = frame.shape
            if self.video_out is None:
                self.video_out = cv.VideoWriter(
                    output_path, cv.VideoWriter_fourcc(*'mp4v'),
                    out_fps, (w, h),
                )
            now_time = time.time()
            if last_frame_time is not None:
                time_diff = now_time - last_frame_time
                while time_diff > 0.0:
                    self.video_out.write(frame)
                    time_diff -= out_frame_interval
            last_frame_time = now_time
        self.video_out.release()
        with self.video_out_done:
            self.video_out_done.notify_all()

    def close_recording(self):
        if self.video_out is not None:
            self.video_out_should_stop = True
            self.video_out_queue.put_nowait(None)
            with self.video_out_done:
                self.video_out_done.wait()
