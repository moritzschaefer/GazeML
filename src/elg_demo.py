#!/usr/bin/env python3
"""Main script for gaze direction inference from webcam feed."""
import argparse
import os
import queue
import time

import numpy as np
import pandas as pd

import coloredlogs
import cv2 as cv
import tensorflow as tf
from datasources import Video, Webcam
from models import DPG, ELG
from util.inference import InferenceThread
from util.video_output import RecordVideoThread
from util.visualization import visualize_forever

if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Demonstration of landmarks localization.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('--from_video', type=str, help='Use this video path instead of webcam')
    parser.add_argument('--record_video', type=str, help='Output path of video of demonstration.')
    parser.add_argument('--record_eye_data', type=str, help='Output path of predicted eye gaze.')
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--headless', action='store_true')

    parser.add_argument('--fps', type=int, default=60, help='Desired sampling rate of webcam')
    parser.add_argument('--camera_id', type=int, default=0, help='ID of webcam to use')

    parser.add_argument('--use_dpg', action='store_true')
    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Check if GPU is available
    from tensorflow.python.client import device_lib
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_available = False
    try:
        gpus = [d for d in device_lib.list_local_devices(config=session_config)
                if d.device_type == 'GPU']
        gpu_available = len(gpus) > 0
    except:
        pass

    # Initialize Tensorflow session
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Session(config=session_config) as session:

        # Declare some parameters
        batch_size = 2

        # Define webcam stream data source
        # Change data_format='NHWC' if not using CUDA
        if args.from_video:
            assert os.path.isfile(args.from_video)
            data_source = Video(args.from_video,
                                tensorflow_session=session, batch_size=batch_size,
                                data_format='NCHW' if gpu_available else 'NHWC',
                                eye_image_shape=(108, 180))
            first_layer_stride = 3
            num_modules = 3
            num_feature_maps = 64
        else:
            data_source = Webcam(tensorflow_session=session, batch_size=batch_size,
                                 camera_id=args.camera_id, fps=args.fps,
                                 data_format='NCHW' if gpu_available else 'NHWC',
                                 eye_image_shape=(36, 60))
            first_layer_stride = 1
            num_modules = 2
            num_feature_maps = 32

        if args.use_dpg:
            model = DPG(session, train_data={'videostream': data_source},
                        learning_schedule=[
                {
                        'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                        },
            ])
        else:
            model = ELG(
                session, train_data={'videostream': data_source},
                first_layer_stride=first_layer_stride,
                num_modules=num_modules,
                num_feature_maps=num_feature_maps,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                    },
                ],
            )
        if args.record_eye_data:
            eye_data = []
        else:
            eye_data = None

        # Record output frames to file if requested
        record_thread = RecordVideoThread(args=(args.record_video, data_source))
        if args.record_video:
            record_thread.daemon = True
            record_thread.start()

        # Start inference
        inferred_stuff_queue = queue.Queue()
        inference_thread = InferenceThread(
            args=(session, model, data_source, inferred_stuff_queue)
        )
        inference_thread.start()

        visualize_forever(
            args,
            inferred_stuff_queue,
            data_source,
            record_thread.video_out_queue,
            batch_size,
            eye_data
        )

        inference_thread.stop()
        inference_thread.join()

        if args.record_video:
            record_thread.close_recording()

        if args.record_eye_data:
            pd.DataFrame(eye_data).to_csv(args.record_eye_data)
