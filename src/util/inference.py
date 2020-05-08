import threading

import numpy as np


class InferenceThread(threading.Thread):
    def __init__(self, target=None, *args, **kwargs):
        super().__init__(target=self._infer, *args, **kwargs)
        self.daemon = True
        self._stop_event = threading.Event()

    def _infer(self, session, model, data_source, inferred_stuff_queue):
        with session.graph.as_default():
            infer = model.inference_generator()
            while True:
                output = next(infer)
                for frame_index in np.unique(output['frame_index']):
                    if frame_index not in data_source._frames:
                        continue
                    frame = data_source._frames[frame_index]
                    if 'inference' in frame['time']:
                        frame['time']['inference'] += output['inference_time']
                    else:
                        frame['time']['inference'] = output['inference_time']
                inferred_stuff_queue.put_nowait(output)

                if self._stop_event.is_set():
                    break

    def stop(self):
        self._stop_event.set()
