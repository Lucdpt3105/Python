"""
BÀI TẬP 2 - YÊU CẦU 1: Phát âm thanh ở nhiều vị trí khác nhau
Play single sound at several positions (x, y, z)
Ví dụ: (0,0,0), (-10,0,0), (5,0,0), (0,5,0), (0,-10,0), (0,0,-10), (0,0,10)
"""

import sys
import os
# Thêm đường dẫn để import openal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from openal import *
import time

class SoundAtPositions:
    def __init__(self):
        print("=== BÀI TẬP 2 - YÊU CẦU 1 ===")
        print("Phát âm thanh ở nhiều vị trí khác nhau trong không gian 3D\n")
        
        # Khởi tạo listener (người nghe) ở giữa
        self.listener = Listener()
        self.listener.position = (0, 0, 0)
        print(f"Listener position: {self.listener.position}")
        
        # Load file âm thanh (đảm bảo có file này trong thư mục)
        # Bạn có thể thay đổi đường dẫn đến file .wav của bạn
        sound_path = os.path.join(os.path.dirname(__file__), 'tone5.wav')
        try:
            self.sound = LoadSound(sound_path)
            print("Đã load âm thanh thành công!\n")
        except Exception as e:
            print(f"Lỗi: Không tìm thấy file âm thanh! {e}")
            print(f"Đường dẫn: {sound_path}")
            print("Vui lòng cung cấp file .wav hoặc điều chỉnh đường dẫn\n")
            self.sound = None
            return
        
        # Tạo player để phát âm thanh
        self.player = Player()
        self.player.add(self.sound)
        
        # Thiết lập các thông số âm thanh
        self.player.rolloff = 0.5  # Độ suy giảm âm thanh theo khoảng cách
        self.player.gain = 1.0     # Độ lớn âm thanh
        
        # Danh sách các vị trí để phát âm thanh
        self.positions = [
            (0, 0, 0),      # Vị trí gốc
            (-10, 0, 0),    # Bên trái
            (10, 0, 0),     # Bên phải
            (0, 10, 0),     # Phía trước
            (0, -10, 0),    # Phía sau
            (5, 0, 0),      # Gần bên phải
            (0, 5, 0),      # Gần phía trước
            (0, 0, 10),     # Phía trên
            (0, 0, -10),    # Phía dưới
        ]
        
    def play_at_positions(self):
        """Phát âm thanh lần lượt ở từng vị trí"""
        print("Bắt đầu phát âm thanh ở các vị trí khác nhau:")
        print("-" * 50)
        
        for i, pos in enumerate(self.positions, 1):
            print(f"\n{i}. Đang phát âm thanh tại vị trí: {pos}")
            print(f"   Khoảng cách từ listener: {self._calculate_distance(pos):.2f}")
            
            # Đặt vị trí cho player
            self.player.position = pos
            
            # Phát âm thanh
            self.player.play()
            
            # Chờ 2 giây để nghe rõ
            time.sleep(2)
            
            # Dừng trước khi chuyển sang vị trí khác
            self.player.stop()
            time.sleep(0.5)
        
        print("\n" + "-" * 50)
        print("Hoàn thành! Đã phát âm thanh ở tất cả các vị trí.")
    
    def _calculate_distance(self, pos):
        """Tính khoảng cách từ listener đến vị trí"""
        import math
        listener_pos = self.listener.position
        return math.sqrt(
            (pos[0] - listener_pos[0])**2 + 
            (pos[1] - listener_pos[1])**2 + 
            (pos[2] - listener_pos[2])**2
        )
    
    def cleanup(self):
        """Giải phóng tài nguyên"""
        try:
            self.player.delete()
            self.sound.delete()
            self.listener.delete()
            print("\nĐã giải phóng tài nguyên.")
        except:
            pass

# Chạy chương trình
if __name__ == "__main__":
    example = SoundAtPositions()
    try:
        example.play_at_positions()
    except KeyboardInterrupt:
        print("\n\nĐã dừng chương trình.")
    finally:
        example.cleanup()
