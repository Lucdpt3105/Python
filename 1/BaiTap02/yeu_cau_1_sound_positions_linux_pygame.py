import math
import os
import sys
import time

import pygame


class SoundAtPositionsLinuxPygame:
    def __init__(self):
        print("=== BAI TAP 2 - YEU CAU 1 (Linux + pygame) ===")
        print("Phat am thanh o nhieu vi tri khac nhau trong khong gian mo phong 3D\n")

        # Listener co dinh tai goc toa do.
        self.listener = (0.0, 0.0, 0.0)
        print(f"Listener position: {self.listener}")

        # Giu danh sach vi tri tu bai goc.
        self.positions = [
            (0, 0, 0),
            (-10, 0, 0),
            (10, 0, 0),
            (0, 10, 0),
            (0, -10, 0),
            (5, 0, 0),
            (0, 5, 0),
            (0, 0, 10),
            (0, 0, -10),
        ]

        # Khoi tao mixer stereo de dieu chinh left/right volume.
        try:
            pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=1024)
            pygame.init()
            pygame.mixer.init()
        except pygame.error as exc:
            raise RuntimeError(
                "Khong khoi tao duoc pygame mixer. "
                "Kiem tra thiet bi am thanh tren Linux (PulseAudio/PipeWire/ALSA)."
            ) from exc

        sound_path = os.path.join(os.path.dirname(__file__), "tone5.wav")
        if not os.path.exists(sound_path):
            raise FileNotFoundError(
                f"Khong tim thay file am thanh: {sound_path}\n"
                "Hay dat tone5.wav cung thu muc voi file nay."
            )

        self.sound = pygame.mixer.Sound(sound_path)
        print("Da load am thanh thanh cong!\n")

    def _distance(self, pos):
        dx = pos[0] - self.listener[0]
        dy = pos[1] - self.listener[1]
        dz = pos[2] - self.listener[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _calculate_pan_and_volume(self, pos):
        """
        Tra ve (left_volume, right_volume) trong khoang [0, 1].

        Mo phong:
        - Pan theo x: x am nghieng trai, x duong nghieng phai.
        - Distance attenuation: giam am theo khoang cach 3D.
        """
        # Distance rolloff: 1 / (1 + k * d)
        distance = self._distance(pos)
        rolloff = 0.15
        base_volume = 1.0 / (1.0 + rolloff * distance)

        # Pan trong [-1, 1] dua theo x.
        max_pan_x = 10.0
        pan = max(-1.0, min(1.0, pos[0] / max_pan_x))

        # Chuyen pan -> he so am luong trai/phai.
        left = base_volume * (1.0 - pan) * 0.5
        right = base_volume * (1.0 + pan) * 0.5

        return left, right, distance

    def play_at_positions(self, hold_seconds=2.0, gap_seconds=0.5):
        print("Bat dau phat am thanh o cac vi tri khac nhau:")
        print("-" * 60)

        for index, pos in enumerate(self.positions, start=1):
            left, right, distance = self._calculate_pan_and_volume(pos)
            print(f"\n{index}. Vi tri: {pos}")
            print(f"   Khoang cach: {distance:.2f}")
            print(f"   Left/Right volume: {left:.3f}/{right:.3f}")

            channel = self.sound.play()
            if channel is None:
                print("   Canh bao: Khong lay duoc kenh phat am thanh.")
                continue

            channel.set_volume(left, right)
            time.sleep(hold_seconds)
            channel.stop()
            time.sleep(gap_seconds)

        print("\n" + "-" * 60)
        print("Hoan thanh! Da phat am thanh tai tat ca vi tri.")

    def cleanup(self):
        pygame.mixer.quit()
        pygame.quit()


def main():
    app = None
    try:
        app = SoundAtPositionsLinuxPygame()
        app.play_at_positions()
    except KeyboardInterrupt:
        print("\n\nDa dung chuong trinh.")
    except Exception as exc:
        print(f"\nLoi: {exc}")
        sys.exit(1)
    finally:
        if app is not None:
            app.cleanup()


if __name__ == "__main__":
    main()
