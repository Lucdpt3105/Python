"""
BÀI TẬP 2 - YÊU CẦU 2: Mô phỏng không gian ảo 3D
Simulate 1 virtual space (eg, a room, a forest area...) which has many sound 
sources at different positions playing their own voice (the sound sources can be moving).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from openal import *
import time
import math

class VirtualSpace3D:
    def __init__(self):
        print("=== BÀI TẬP 2 - YÊU CẦU 2 ===")
        print("Mô phỏng không gian ảo 3D với nhiều nguồn âm thanh di chuyển\n")
        
        # Khởi tạo listener (người nghe) ở giữa phòng
        self.listener = Listener()
        self.listener.position = (0, 0, 0)
        print(f"Listener (bạn) đang đứng ở vị trí: {self.listener.position}")
        print("Đây là trung tâm của một căn phòng 20x20x10 mét\n")
        
        # Load nhiều file âm thanh khác nhau
        self.sounds = {}
        sound_files = {
            'bird1': 'bird1.wav',
            'bird2': 'bird2.wav',
            'wind': 'wind.wav',
            'animal': 'animal.wav',
            'ambient': 'tone5.wav'  # Dùng tone5 cho ambient
        }
        
        print("Đang load các file âm thanh...")
        sounds_dir = os.path.join(os.path.dirname(__file__), 'sounds')
        tone5_path = os.path.join(os.path.dirname(__file__), '..', '3D_Audio', 'tone5.wav')
        
        for key, filename in sound_files.items():
            try:
                if key == 'ambient':
                    # Ambient dùng tone5 từ thư mục 3D_Audio
                    self.sounds[key] = LoadSound(tone5_path)
                else:
                    # Các âm thanh khác từ thư mục sounds
                    sound_path = os.path.join(sounds_dir, filename)
                    self.sounds[key] = LoadSound(sound_path)
                print(f"  ✓ {key}: {filename}")
            except Exception as e:
                print(f"  ✗ Lỗi load {key}: {e}")
                return
        
        print("Đã load tất cả âm thanh thành công!\n")
        
        # Tạo nhiều nguồn âm thanh (sound sources)
        self.sources = []
        self.create_sound_sources()
        
        print(f"\nĐã tạo {len(self.sources)} nguồn âm thanh trong không gian 3D")
        
    def create_sound_sources(self):
        """Tạo nhiều nguồn âm thanh ở các vị trí khác nhau"""
        
        # Cấu hình các nguồn âm thanh với vị trí ban đầu và hướng di chuyển
        source_configs = [
            {
                'name': 'Bird 1 (Chim 1)',
                'sound_key': 'bird1',
                'start_pos': (-8, 5, 3),
                'direction': (1, 0, 0),    # Di chuyển sang phải
                'speed': 0.3,
                'gain': 0.8,
                'pitch': 1.2
            },
            {
                'name': 'Bird 2 (Chim 2)',
                'sound_key': 'bird2',
                'start_pos': (8, -5, 4),
                'direction': (-1, 0, 0),   # Di chuyển sang trái
                'speed': 0.4,
                'gain': 0.7,
                'pitch': 1.5
            },
            {
                'name': 'Wind (Gió)',
                'sound_key': 'wind',
                'start_pos': (0, 10, 0),
                'direction': (0, -1, 0),   # Di chuyển xuống
                'speed': 0.2,
                'gain': 0.5,
                'pitch': 0.8
            },
            {
                'name': 'Animal (Động vật)',
                'sound_key': 'animal',
                'start_pos': (0, -10, 2),
                'direction': (0, 1, 0),    # Di chuyển lên
                'speed': 0.25,
                'gain': 0.9,
                'pitch': 0.9
            },
            {
                'name': 'Circular Sound (Âm thanh xoay tròn)',
                'sound_key': 'ambient',
                'start_pos': (7, 0, 2),
                'direction': (0, 1, 0),    # Sẽ được tính toán để xoay tròn
                'speed': 0.5,
                'gain': 0.6,
                'pitch': 1.1,
                'circular': True
            }
        ]
        
        for config in source_configs:
            player = Player()
            # Load âm thanh riêng cho từng nguồn
            sound = self.sounds[config['sound_key']]
            player.add(sound)
            player.position = config['start_pos']
            player.rolloff = 0.3
            player.gain = config['gain']
            player.pitch = config['pitch']
            player.loop = True  # Loop âm thanh để phát liên tục
            
            self.sources.append({
                'player': player,
                'name': config['name'],
                'position': list(config['start_pos']),
                'direction': config['direction'],
                'speed': config['speed'],
                'circular': config.get('circular', False),
                'angle': 0  # Cho chuyển động xoay tròn
            })
            
            print(f"  + {config['name']}: vị trí {config['start_pos']}")
    
    def simulate_space(self, duration=30):
        """
        Mô phỏng không gian với các nguồn âm thanh di chuyển
        duration: thời gian mô phỏng (giây)
        """
        print(f"\n{'='*60}")
        print("BẮT ĐẦU MÔ PHỎNG KHÔNG GIAN 3D")
        print(f"Thời gian: {duration} giây")
        print(f"{'='*60}\n")
        
        # Bắt đầu phát tất cả các nguồn âm thanh
        for source in self.sources:
            source['player'].play()
        
        print("Tất cả nguồn âm thanh đang phát và di chuyển...\n")
        
        # Mô phỏng chuyển động
        steps = duration * 10  # 10 bước mỗi giây
        for step in range(steps):
            time.sleep(0.1)  # 100ms mỗi bước
            
            # Cập nhật vị trí của từng nguồn âm thanh
            for source in self.sources:
                self._update_source_position(source)
            
            # Hiển thị thông tin mỗi 2 giây
            if step % 20 == 0:
                elapsed = step / 10
                print(f"[{elapsed:.1f}s] Vị trí các nguồn âm thanh:")
                for source in self.sources:
                    pos = source['position']
                    distance = self._calculate_distance(pos)
                    print(f"  • {source['name']}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) "
                          f"- Khoảng cách: {distance:.1f}m")
                print()
        
        print(f"{'='*60}")
        print("KẾT THÚC MÔ PHỎNG")
        print(f"{'='*60}\n")
        
        # Dừng tất cả nguồn âm thanh
        for source in self.sources:
            source['player'].stop()
    
    def _update_source_position(self, source):
        """Cập nhật vị trí của nguồn âm thanh"""
        if source.get('circular', False):
            # Chuyển động xoay tròn
            source['angle'] += source['speed'] * 0.1
            radius = 7
            source['position'][0] = radius * math.cos(source['angle'])
            source['position'][1] = radius * math.sin(source['angle'])
        else:
            # Chuyển động thẳng
            for i in range(3):
                source['position'][i] += source['direction'][i] * source['speed']
            
            # Giới hạn trong không gian phòng và đảo chiều khi chạm tường
            limits = [(-10, 10), (-10, 10), (0, 8)]
            for i in range(3):
                if source['position'][i] < limits[i][0]:
                    source['position'][i] = limits[i][0]
                    source['direction'] = list(source['direction'])
                    source['direction'][i] *= -1
                    source['direction'] = tuple(source['direction'])
                elif source['position'][i] > limits[i][1]:
                    source['position'][i] = limits[i][1]
                    source['direction'] = list(source['direction'])
                    source['direction'][i] *= -1
                    source['direction'] = tuple(source['direction'])
        
        # Cập nhật vị trí trong OpenAL
        source['player'].position = tuple(source['position'])
    
    def _calculate_distance(self, pos):
        """Tính khoảng cách từ listener"""
        return math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    
    def cleanup(self):
        """Giải phóng tài nguyên"""
        print("Đang dọn dẹp tài nguyên...")
        try:
            for source in self.sources:
                source['player'].delete()
            for sound in self.sounds.values():
                sound.delete()
            self.listener.delete()
            print("Đã giải phóng tài nguyên.\n")
        except:
            pass

# Chạy chương trình
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MÔ PHỎNG KHÔNG GIAN ẢO 3D - OPENAL")
    print("="*60 + "\n")
    
    space = VirtualSpace3D()
    
    try:
        # Chạy mô phỏng trong 30 giây (có thể thay đổi)
        space.simulate_space(duration=30)
        
    except KeyboardInterrupt:
        print("\n\nĐã dừng mô phỏng bằng Ctrl+C.")
        
    finally:
        space.cleanup()
        print("Chương trình kết thúc.")
