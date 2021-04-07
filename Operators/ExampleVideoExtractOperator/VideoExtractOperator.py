import cv2

from Operators.DummyOperator import DummyOperator
from Utils.Exceptions import VideoExtractMethodNotSupportException
from Utils.Storage.BaseOSS import CloudObjectStorage
from concurrent.futures import ThreadPoolExecutor

from Utils.misc import get_date_string, get_uuid_name
import os


class VideoExtractOperator(DummyOperator):
    name = '视频关键帧提取'
    __version__ = 'v1.0.20210326'

    def __init__(self, _is_test):
        super().__init__(_is_test)

    def execute(self, _video_url, _extract_mode, _interval_count,
                _oss_helper: CloudObjectStorage,
                _target_bucket=None,
                ):
        """
        进行视频关键帧提取，并存在在oss中

        Args:
            _video_url:     视频地址
            _extract_mode:  提取模式
            _interval_count:    提取间隔
            _oss_helper:    oss
            _target_bucket:     目标bucket name

        Returns:    每个关键帧的bucket name和path

        """
        cap = cv2.VideoCapture(_video_url)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if _extract_mode == 'n_seconds':
            interval_frames = int(video_fps * _interval_count)
        elif _extract_mode == 'n_frames':
            interval_frames = _interval_count
        else:
            raise VideoExtractMethodNotSupportException(f'{_extract_mode} not support now')
        video_frame_position = 0
        date_string = get_date_string()
        uuid_name = get_uuid_name()
        all_tasks = []
        with ThreadPoolExecutor() as executor:
            while True:
                grabbed, m_frame = cap.read()
                if grabbed:
                    if _oss_helper and _target_bucket:
                        m_frame_target_bucket = _target_bucket
                        m_frame_target_path = os.path.join(date_string, uuid_name, f'{video_frame_position}')
                        all_tasks.append(executor.submit(
                            _oss_helper.upload_image_file,
                            m_frame_target_bucket,
                            m_frame_target_path,
                            m_frame, False)
                        )
                    video_frame_position += interval_frames
                    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_position)
                else:
                    break
        cap.release()
        # 严格保证按顺序返回任务
        for m_task in all_tasks:
            if m_task.done():
                # 这里原本设计是可以返回frame信息的，但考虑到整体架构，这样反而会降低程序的并行度，
                # 提升单节点的耗时，所以将frame的返回直接去掉了。
                yield _target_bucket, m_task.result()


if __name__ == '__main__':
    from Utils.Storage import DummyOSS
    from argparse import ArgumentParser

    ag = ArgumentParser('Video Frames Extract Example')
    ag.add_argument('-v', '--video', type=str, dest='video',
                    default='http://vfx.mtime.cn/Video/2019/02/04/mp4/190204084208765161.mp4',
                    help='待提取视频')
    ag.add_argument('-m', '--mode', type=str, dest='mode', choices=['n_frames', 'n_seconds'],
                    default='n_frames',
                    help='视频关键帧提取方式，（n_frames：按间隔N帧提取（默认）;n_seconds：按间隔N秒提取）'
                    )
    ag.add_argument('-i', '--interval', type=int, dest='interval', required=True, help='间隔N值')

    args = ag.parse_args()
    video_extractor = VideoExtractOperator(True)
    oss_helper = DummyOSS(None, None, None)
    all_frame_info = video_extractor.execute(args.video,
                                             args.mode, args.interval,
                                             oss_helper,
                                             'downloaded_image_frames')
    for m_frame_bucket, m_frame_path in all_frame_info:
        print(m_frame_bucket, m_frame_path)
        print(oss_helper.get_retrieve_url(m_frame_bucket, m_frame_path, 86400))
        print('*' * 30)
