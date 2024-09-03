import fractions
import logging
import math
from itertools import tee
from struct import pack, unpack_from
from typing import Iterator, List, Optional, Sequence, Tuple, Type, TypeVar

import av
from av.frame import Frame
from av.packet import Packet

from ..jitterbuffer import JitterFrame
from ..mediastreams import VIDEO_TIME_BASE, convert_timebase
from .base import Decoder, Encoder

logger = logging.getLogger(__name__)

Bitrate_constant = 6000000
DEFAULT_BITRATE = Bitrate_constant
MIN_BITRATE = Bitrate_constant
MAX_BITRATE = Bitrate_constant

MAX_FRAME_RATE = 30
PACKET_MAX = 1300

NAL_TYPE_FU_A = 28
NAL_TYPE_STAP_A = 24

NAL_HEADER_SIZE = 1
FU_A_HEADER_SIZE = 2
LENGTH_FIELD_SIZE = 2
STAP_A_HEADER_SIZE = NAL_HEADER_SIZE + LENGTH_FIELD_SIZE

DESCRIPTOR_T = TypeVar("DESCRIPTOR_T", bound="H264PayloadDescriptor")
T = TypeVar("T")


def pairwise(iterable: Sequence[T]) -> Iterator[Tuple[T, T]]:
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class H264PayloadDescriptor:
    def __init__(self, first_fragment):
        self.first_fragment = first_fragment

    def __repr__(self):
        return f"H264PayloadDescriptor(FF={self.first_fragment})"

    @classmethod
    def parse(cls: Type[DESCRIPTOR_T], data: bytes) -> Tuple[DESCRIPTOR_T, bytes]:
        output = bytes()

        # NAL unit header
        if len(data) < 2:
            raise ValueError("NAL unit is too short")
        nal_type = data[0] & 0x1F
        f_nri = data[0] & (0x80 | 0x60)
        pos = NAL_HEADER_SIZE

        if nal_type in range(1, 24):
            # single NAL unit
            output = bytes([0, 0, 0, 1]) + data
            obj = cls(first_fragment=True)
        elif nal_type == NAL_TYPE_FU_A:
            # fragmentation unit
            original_nal_type = data[pos] & 0x1F
            first_fragment = bool(data[pos] & 0x80)
            pos += 1

            if first_fragment:
                original_nal_header = bytes([f_nri | original_nal_type])
                output += bytes([0, 0, 0, 1])
                output += original_nal_header
            output += data[pos:]

            obj = cls(first_fragment=first_fragment)
        elif nal_type == NAL_TYPE_STAP_A:
            # single time aggregation packet
            offsets = []
            while pos < len(data):
                if len(data) < pos + LENGTH_FIELD_SIZE:
                    raise ValueError("STAP-A length field is truncated")
                nalu_size = unpack_from("!H", data, pos)[0]
                pos += LENGTH_FIELD_SIZE
                offsets.append(pos)

                pos += nalu_size
                if len(data) < pos:
                    raise ValueError("STAP-A data is truncated")

            offsets.append(len(data) + LENGTH_FIELD_SIZE)
            for start, end in pairwise(offsets):
                end -= LENGTH_FIELD_SIZE
                output += bytes([0, 0, 0, 1])
                output += data[start:end]

            obj = cls(first_fragment=True)
        else:
            raise ValueError(f"NAL unit type {nal_type} is not supported")

        return obj, output


# class H264Decoder(Decoder):
#     def __init__(self) -> None:
#         self.codec = av.CodecContext.create("h264", "r")
#
#     def decode(self, encoded_frame: JitterFrame) -> List[Frame]:
#         try:
#             packet = av.Packet(encoded_frame.data)
#             packet.pts = encoded_frame.timestamp
#             packet.time_base = VIDEO_TIME_BASE
#             frames = self.codec.decode(packet)
#         except av.AVError as e:
#             logger.warning(
#                 "H264Decoder() failed to decode, skipping package: " + str(e)
#             )
#             return []
#
#         return frames

class H264Decoder(Decoder):
    def _init_(self) -> None:
        self.codec = av.CodecContext.create("h264", "r")
        # self.gop_size = self.codec.gop_size
        self.frame_count = 0  # Add this line to initialize the frame count
        # self.frame_file_count = 1
        # self.start_time = time.time()

    def decode(self, encoded_frame: JitterFrame) -> List[Frame]:
        try:
            packet = av.Packet(encoded_frame.data)
            packet.pts = encoded_frame.timestamp
            packet.time_base = VIDEO_TIME_BASE
            frames = self.codec.decode(packet)
            # For calculating FPS manually:
            # self.frame_count += 1
            # elapsed_time = time.time() - self.start_time
            # if elapsed_time > 1:  # Check every second
            #     fps = self.frame_count / elapsed_time
            #     print(f"FPS: {fps:.2f}")
            #     self.frame_count = 0
            #     self.start_time = time.time()
            # # For calculating FPS manually:
            #
            # # For calculating GOP manually:
            for frame in frames:
                # img_array = frame.to_ndarray(format="bgr24")
                if frame.pict_type == av.video.frame.PictureType.I:  # Check if the frame is an I-frame
                    print(f"The GOP size is: {self.frame_count}")
                    # filename = f"screendecode/{self.frame_file_count:04}_i.png"
                    # self.frame_file_count += 1
                    self.frame_count = 0  # Reset the frame count after finding an I-frame
                else:
                    self.frame_count += 1
                    # filename = f"screendecode/{self.frame_file_count:04}.png"
                    # self.frame_file_count += 1
                # cv2.imwrite(filename, img_array)
            # For calculating GOP manually:
        except av.AVError as e:
            logger.warning(
                "H264Decoder() failed to decode, skipping package: " + str(e)
            )
            return []

        return frames


def create_encoder_context(
    codec_name: str, width: int, height: int, bitrate: int
) -> Tuple[av.CodecContext, bool]:
    codec = av.CodecContext.create(codec_name, "w")
    codec.width = width
    codec.height = height
    codec.bit_rate = bitrate
    codec.pix_fmt = "yuv420p"
    # Framerate and time base
    codec.framerate = fractions.Fraction(MAX_FRAME_RATE, 1)
    codec.time_base = fractions.Fraction(1, MAX_FRAME_RATE)

    # GOP size for 1.5-second keyframe interval at MAX_FRAME_RATE
    # codec.gop_size = 1.5 * MAX_FRAME_RATE
    # codec.options = {
    #     "profile": "high",
    #     "level": "4.2" if MAX_FRAME_RATE == 60 else "4.0",
    #     "tune": "zerolatency",
    #     "preset": "ultrafast",
    #     "bframes": "0",
    #     # "ssim": "true",  # SSIM calculation for quality check
    #     # "psnr": "true",  # PSNR calculation for quality check
    # }
    # codec.options = {
    #     "profile": "baseline",
    #     "level": "31",
    #     "tune": "zerolatency",  # does nothing using h264_omx
    # }
    # print("codec bitrate", bitrate)
    # codec.options = {
    #     # 'profile': 'baseline',
    #     'preset': 'ultrafast',  # Fastest encoding with reduced delay
    #     'tune': 'zerolatency',  # Reduce encoder latency
    #     # "level": "31",
    #     'x264opts': 'nal-hrd=cbr:force-cfr=1',
    #     'bf': '0',  # Disable B-frames to reduce complexity and latency
    #     'g': '30',
    #     'refs': '1',
    #     'rc-lookahead': '0',
    #     'threads': 'auto',
    #     # 'slice-max-size': '1300',
    #     'nal-hrd': 'cbr',
    #     'force-cfr': '1',
    #     # 'vbv-bufsize': str(bitrate // MAX_FRAME_RATE),  # Set buffer size to one second of video at the target bitrate
    #     'vbv-bufsize': str(bitrate //  MAX_FRAME_RATE),  # Set buffer size to one second of video at the target bitrate
    #     'vbv-maxrate': str(bitrate),
    # }
    codec.options = {
        'preset': 'ultrafast',  # Use 'ultrafast' for minimal encoding delay
        'tune': 'zerolatency',  # Tune for zero latency
        'g': '45',  # GOP size
        'refs': '1',  # Reference frames
        'rc-lookahead': '0',  # Lookahead frames for rate control
        # 'threads': 'auto',  # Use automatic threading
        'nal-hrd': 'cbr',  # Constant Bitrate mode
        'force-cfr': '1',  # Force constant framerate
        'vbv-bufsize': str(bitrate // MAX_FRAME_RATE),  # Buffer size for the rate control
        'vbv-maxrate': str(bitrate // MAX_FRAME_RATE),  # Maximum bitrate
        'rc': 'cbr',  # Constant bitrate, low-delay high quality
        'zerolatency': '1',  # Enable zero latency
        'forced-idr': '0',  # Force IDR frames
        # 'pix_fmt': 'yuv420p',  # Pixel format
        'b:v': f'{bitrate}',  # Bitrate
        'minrate': f'{bitrate}',  # Minimum bitrate
        'maxrate': f'{bitrate}',  # Maximum bitrate
        'bufsize': f'{bitrate // MAX_FRAME_RATE}',  # Buffer size
    }

    codec.open()
    return codec, codec_name == "libx264"


class H264Encoder(Encoder):
    def __init__(self) -> None:
        self.buffer_data = b""
        self.buffer_pts: Optional[int] = None
        self.codec: Optional[av.CodecContext] = None
        self.codec_buffering = False
        self.__target_bitrate = DEFAULT_BITRATE

    @staticmethod
    def _packetize_fu_a(data: bytes) -> List[bytes]:
        available_size = PACKET_MAX - FU_A_HEADER_SIZE
        payload_size = len(data) - NAL_HEADER_SIZE
        num_packets = math.ceil(payload_size / available_size)
        num_larger_packets = payload_size % num_packets
        package_size = payload_size // num_packets

        f_nri = data[0] & (0x80 | 0x60)  # fni of original header
        nal = data[0] & 0x1F

        fu_indicator = f_nri | NAL_TYPE_FU_A

        fu_header_end = bytes([fu_indicator, nal | 0x40])
        fu_header_middle = bytes([fu_indicator, nal])
        fu_header_start = bytes([fu_indicator, nal | 0x80])
        fu_header = fu_header_start

        packages = []
        offset = NAL_HEADER_SIZE
        while offset < len(data):
            if num_larger_packets > 0:
                num_larger_packets -= 1
                payload = data[offset : offset + package_size + 1]
                offset += package_size + 1
            else:
                payload = data[offset : offset + package_size]
                offset += package_size

            if offset == len(data):
                fu_header = fu_header_end

            packages.append(fu_header + payload)

            fu_header = fu_header_middle
        assert offset == len(data), "incorrect fragment data"

        return packages

    @staticmethod
    def _packetize_stap_a(
        data: bytes, packages_iterator: Iterator[bytes]
    ) -> Tuple[bytes, bytes]:
        counter = 0
        available_size = PACKET_MAX - STAP_A_HEADER_SIZE

        stap_header = NAL_TYPE_STAP_A | (data[0] & 0xE0)

        payload = bytes()
        try:
            nalu = data  # with header
            while len(nalu) <= available_size and counter < 9:
                stap_header |= nalu[0] & 0x80

                nri = nalu[0] & 0x60
                if stap_header & 0x60 < nri:
                    stap_header = stap_header & 0x9F | nri

                available_size -= LENGTH_FIELD_SIZE + len(nalu)
                counter += 1
                payload += pack("!H", len(nalu)) + nalu
                nalu = next(packages_iterator)

            if counter == 0:
                nalu = next(packages_iterator)
        except StopIteration:
            nalu = None

        if counter <= 1:
            return data, nalu
        else:
            return bytes([stap_header]) + payload, nalu

    @staticmethod
    def _split_bitstream(buf: bytes) -> Iterator[bytes]:
        # Translated from: https://github.com/aizvorski/h264bitstream/blob/master/h264_nal.c#L134
        i = 0
        while True:
            # Find the start of the NAL unit.
            #
            # NAL Units start with the 3-byte start code 0x000001 or
            # the 4-byte start code 0x00000001.
            i = buf.find(b"\x00\x00\x01", i)
            if i == -1:
                return

            # Jump past the start code
            i += 3
            nal_start = i

            # Find the end of the NAL unit (end of buffer OR next start code)
            i = buf.find(b"\x00\x00\x01", i)
            if i == -1:
                yield buf[nal_start : len(buf)]
                return
            elif buf[i - 1] == 0:
                # 4-byte start code case, jump back one byte
                yield buf[nal_start : i - 1]
            else:
                yield buf[nal_start:i]

    @classmethod
    def _packetize(cls, packages: Iterator[bytes]) -> List[bytes]:
        packetized_packages = []

        packages_iterator = iter(packages)
        package = next(packages_iterator, None)
        while package is not None:
            if len(package) > PACKET_MAX:
                packetized_packages.extend(cls._packetize_fu_a(package))
                package = next(packages_iterator, None)
            else:
                packetized, package = cls._packetize_stap_a(package, packages_iterator)
                packetized_packages.append(packetized)

        return packetized_packages

    def _encode_frame(
        self, frame: av.VideoFrame, force_keyframe: bool
    ) -> Iterator[bytes]:
        if self.codec and (
            frame.width != self.codec.width
            or frame.height != self.codec.height
            # we only adjust bitrate if it changes by over 10%
            or abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate
            > 0.1
        ):
            self.buffer_data = b""
            self.buffer_pts = None
            self.codec = None

        if force_keyframe:
            # force a complete image
            frame.pict_type = av.video.frame.PictureType.I
        else:
            # reset the picture type, otherwise no B-frames are produced
            frame.pict_type = av.video.frame.PictureType.NONE

        if self.codec is None:
            try:
                self.codec, self.codec_buffering = create_encoder_context(
                    "libx264", frame.width, frame.height, bitrate=self.target_bitrate
                )
            except Exception:
                self.codec, self.codec_buffering = create_encoder_context(
                    "libx264",
                    frame.width,
                    frame.height,
                    bitrate=self.target_bitrate,
                )

        data_to_send = b""
        for package in self.codec.encode(frame):
            package_bytes = bytes(package)
            if self.codec_buffering:
                # delay sending to ensure we accumulate all packages
                # for a given PTS
                if package.pts == self.buffer_pts:
                    self.buffer_data += package_bytes
                else:
                    data_to_send += self.buffer_data
                    self.buffer_data = package_bytes
                    self.buffer_pts = package.pts
            else:
                data_to_send += package_bytes

        if data_to_send:
            yield from self._split_bitstream(data_to_send)

    def encode(
        self, frame: Frame, force_keyframe: bool = False
    ) -> Tuple[List[bytes], int]:
        assert isinstance(frame, av.VideoFrame)
        packages = self._encode_frame(frame, force_keyframe)
        timestamp = convert_timebase(frame.pts, frame.time_base, VIDEO_TIME_BASE)
        return self._packetize(packages), timestamp

    def pack(self, packet: Packet) -> Tuple[List[bytes], int]:
        assert isinstance(packet, av.Packet)
        packages = self._split_bitstream(bytes(packet))
        timestamp = convert_timebase(packet.pts, packet.time_base, VIDEO_TIME_BASE)
        return self._packetize(packages), timestamp

    @property
    def target_bitrate(self) -> int:
        """
        Target bitrate in bits per second.
        """
        return self.__target_bitrate

    @target_bitrate.setter
    def target_bitrate(self, bitrate: int) -> None:
        bitrate = max(MIN_BITRATE, min(bitrate, MAX_BITRATE))
        self.__target_bitrate = bitrate


def h264_depayload(payload: bytes) -> bytes:
    descriptor, data = H264PayloadDescriptor.parse(payload)
    return data
