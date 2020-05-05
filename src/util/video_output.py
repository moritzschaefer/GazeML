import queue
import sys
import threading
import time

import cv2 as cv
import numpy as np
import pandas as pd

import util.gaze


def visualize_forever(args, inferred_stuff_queue, data_source, video_out_queue,
                      batch_size, eye_data):
    last_frame_index = 0
    last_frame_time = time.time()
    fps_history = []
    all_gaze_histories = []

    if args.fullscreen:
        cv.namedWindow('vis', cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty('vis', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    while True:
        # If no output to visualize, show unannotated frame
        if inferred_stuff_queue.empty():
            next_frame_index = last_frame_index + 1
            if next_frame_index in data_source._frames:
                next_frame = data_source._frames[next_frame_index]
                if 'faces' in next_frame and len(next_frame['faces']) == 0:
                    if not args.headless:
                        cv.imshow('vis', next_frame['bgr'])
                    if args.record_video:
                        video_out_queue.put_nowait(next_frame_index)
                    last_frame_index = next_frame_index
            if cv.waitKey(1) & 0xFF == ord('q'):
                return
            continue

        # Get output from neural network and visualize
        output = inferred_stuff_queue.get()
        bgr = None
        for j in range(batch_size):
            frame_index = output['frame_index'][j]
            if frame_index not in data_source._frames:
                continue
            frame = data_source._frames[frame_index]

            # Decide which landmarks are usable
            heatmaps_amax = np.amax(output['heatmaps'][j, :].reshape(-1, 18), axis=0)
            can_use_eye = np.all(heatmaps_amax > 0.7)
            can_use_eyelid = np.all(heatmaps_amax[0:8] > 0.75)
            can_use_iris = np.all(heatmaps_amax[8:16] > 0.8)

            start_time = time.time()
            eye_index = output['eye_index'][j]
            bgr = frame['bgr']
            eye = frame['eyes'][eye_index]
            eye_image = eye['image']
            eye_side = eye['side']
            eye_landmarks = output['landmarks'][j, :]
            eye_radius = output['radius'][j][0]
            if eye_side == 'left':
                eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
                eye_image = np.fliplr(eye_image)

            # Embed eye image and annotate for picture-in-picture
            eye_upscale = 2
            eye_image_raw = cv.cvtColor(cv.equalizeHist(eye_image), cv.COLOR_GRAY2BGR)
            eye_image_raw = cv.resize(eye_image_raw, (0, 0), fx=eye_upscale, fy=eye_upscale)
            eye_image_annotated = np.copy(eye_image_raw)
            if can_use_eyelid:
                cv.polylines(
                    eye_image_annotated,
                    [np.round(eye_upscale*eye_landmarks[0:8]).astype(np.int32)
                     .reshape(-1, 1, 2)],
                    isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv.LINE_AA,
                )
            if can_use_iris:
                cv.polylines(
                    eye_image_annotated,
                    [np.round(eye_upscale*eye_landmarks[8:16]).astype(np.int32)
                     .reshape(-1, 1, 2)],
                    isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,
                )
                cv.drawMarker(
                    eye_image_annotated,
                    tuple(np.round(eye_upscale*eye_landmarks[16, :]).astype(np.int32)),
                    color=(0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=4,
                    thickness=1, line_type=cv.LINE_AA,
                )
            face_index = int(eye_index / 2)
            eh, ew, _ = eye_image_raw.shape
            v0 = face_index * 2 * eh
            v1 = v0 + eh
            v2 = v1 + eh
            u0 = 0 if eye_side == 'left' else ew
            u1 = u0 + ew
            bgr[v0:v1, u0:u1] = eye_image_raw
            bgr[v1:v2, u0:u1] = eye_image_annotated

            # Visualize preprocessing results
            frame_landmarks = (frame['smoothed_landmarks']
                               if 'smoothed_landmarks' in frame
                               else frame['landmarks'])
            for f, face in enumerate(frame['faces']):
                for landmark in frame_landmarks[f][:-1]:
                    cv.drawMarker(bgr, tuple(np.round(landmark).astype(np.int32)),
                                  color=(0, 0, 255), markerType=cv.MARKER_STAR,
                                  markerSize=2, thickness=1, line_type=cv.LINE_AA)
                cv.rectangle(
                    bgr, tuple(np.round(face[:2]).astype(np.int32)),
                    tuple(np.round(np.add(face[:2], face[2:])).astype(np.int32)),
                    color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,
                )

            # Transform predictions
            eye_landmarks = np.concatenate([eye_landmarks,
                                            [[eye_landmarks[-1, 0] + eye_radius,
                                                eye_landmarks[-1, 1]]]])
            eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
                                               'constant', constant_values=1.0))
            eye_landmarks = (eye_landmarks *
                             eye['inv_landmarks_transform_mat'].T)[:, :2]
            eye_landmarks = np.asarray(eye_landmarks)
            eyelid_landmarks = eye_landmarks[0:8, :]
            iris_landmarks = eye_landmarks[8:16, :]
            iris_centre = eye_landmarks[16, :]
            eyeball_centre = eye_landmarks[17, :]
            eyeball_radius = np.linalg.norm(eye_landmarks[18, :] -
                                            eye_landmarks[17, :])

            # Smooth and visualize gaze direction
            num_total_eyes_in_frame = len(frame['eyes'])
            if len(all_gaze_histories) != num_total_eyes_in_frame:
                all_gaze_histories = [list() for _ in range(num_total_eyes_in_frame)]
            gaze_history = all_gaze_histories[eye_index]
            if can_use_eye:
                # Visualize landmarks
                cv.drawMarker(  # Eyeball centre
                    bgr, tuple(np.round(eyeball_centre).astype(np.int32)),
                    color=(0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=4,
                    thickness=1, line_type=cv.LINE_AA,
                )
                # cv.circle(  # Eyeball outline
                #     bgr, tuple(np.round(eyeball_centre).astype(np.int32)),
                #     int(np.round(eyeball_radius)), color=(0, 255, 0),
                #     thickness=1, lineType=cv.LINE_AA,
                # )

                # Draw "gaze"
                # from models.elg import estimate_gaze_from_landmarks
                # current_gaze = estimate_gaze_from_landmarks(
                #     iris_landmarks, iris_centre, eyeball_centre, eyeball_radius)
                i_x0, i_y0 = iris_centre
                e_x0, e_y0 = eyeball_centre
                theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
                phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
                                        -1.0, 1.0))
                current_gaze = np.array([theta, phi])
                gaze_history.append(current_gaze)
                gaze_history_max_len = 10
                if len(gaze_history) > gaze_history_max_len:
                    gaze_history = gaze_history[-gaze_history_max_len:]
                util.gaze.draw_gaze(bgr, iris_centre, np.mean(gaze_history, axis=0),
                                    length=120.0, thickness=1)
            else:
                gaze_history.clear()

            if can_use_eyelid:
                cv.polylines(
                    bgr, [np.round(eyelid_landmarks).astype(np.int32).reshape(-1, 1, 2)],
                    isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv.LINE_AA,
                )

            if can_use_iris:
                cv.polylines(
                    bgr, [np.round(iris_landmarks).astype(np.int32).reshape(-1, 1, 2)],
                    isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,
                )
                cv.drawMarker(
                    bgr, tuple(np.round(iris_centre).astype(np.int32)),
                    color=(0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=4,
                    thickness=1, line_type=cv.LINE_AA,
                )

            dtime = 1e3*(time.time() - start_time)
            if 'visualization' not in frame['time']:
                frame['time']['visualization'] = dtime
            else:
                frame['time']['visualization'] += dtime

            def _dtime(before_id, after_id):
                return int(1e3 * (frame['time'][after_id] - frame['time'][before_id]))

            def _dstr(title, before_id, after_id):
                return '%s: %dms' % (title, _dtime(before_id, after_id))

            if eye_index == len(frame['eyes']) - 1:
                # Calculate timings
                frame['time']['after_visualization'] = time.time()
                fps = int(np.round(1.0 / (time.time() - last_frame_time)))
                fps_history.append(fps)
                if len(fps_history) > 60:
                    fps_history = fps_history[-60:]
                fps_str = '%d FPS' % np.mean(fps_history)
                last_frame_time = time.time()
                fh, fw, _ = bgr.shape
                cv.putText(bgr, fps_str, org=(fw - 110, fh - 20),
                           fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8,
                           color=(0, 0, 0), thickness=1, lineType=cv.LINE_AA)
                cv.putText(bgr, fps_str, org=(fw - 111, fh - 21),
                           fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.79,
                           color=(255, 255, 255), thickness=1, lineType=cv.LINE_AA)

                dots = [(100, 100), (50, 200), (200, 200), (200, 50), (300, 100),
                        (20, 200), (20, 20), (400, 200), (200, 400), (600, 400)]
                try:
                    dot = dots[frame_index // 100]
                    cv.circle(bgr, dot, 20, (255, 0, 0), thickness=7)
                except IndexError:
                    pass

                if not args.headless:
                    cv.imshow('vis', bgr)
                last_frame_index = frame_index

                # Record frame?
                if args.record_video:
                    video_out_queue.put_nowait(frame_index)
                if args.record_eye_data and len(frame['eyes']) >= 2 and len(all_gaze_histories) >= 2 and np.all([len(gh) > 0 for gh in all_gaze_histories[:2]]):
                    if frame['eyes'][0]['side'] == 'left':
                        left, right = all_gaze_histories[0][-1], all_gaze_histories[1][-1]
                    else:
                        right, left = all_gaze_histories[0][-1], all_gaze_histories[1][-1]
                    eye_data.append(pd.Series(index=['left_theta', 'left_phi', 'right_theta', 'right_phi'], data=[
                                    *left, *right], name=frame_index))

                # Quit?
                if cv.waitKey(1) & 0xFF == ord('q'):
                    return

                # Print timings
                if frame_index % 60 == 0:
                    latency = _dtime('before_frame_read', 'after_visualization')
                    processing = _dtime('after_frame_read', 'after_visualization')
                    timing_string = ', '.join([
                        _dstr('read', 'before_frame_read', 'after_frame_read'),
                        _dstr('preproc', 'after_frame_read', 'after_preprocessing'),
                        'infer: %dms' % int(frame['time']['inference']),
                        'vis: %dms' % int(frame['time']['visualization']),
                        'proc: %dms' % processing,
                        'latency: %dms' % latency,
                    ])
                    print('%08d [%s] %s' %
                          (frame_index, fps_str, timing_string), file=sys.stderr)


def start_visualize_thread(*args):
    visualize_thread = threading.Thread(
        name='visualization',
        target=visualize_forever,
        args=args,
    )
    visualize_thread.daemon = True
    visualize_thread.start()

    return visualize_thread


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
