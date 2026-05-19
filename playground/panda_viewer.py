# ==============================================================================
# DOSYA: panda_viewer.py
# AÇIKLAMA: PPO Dogfight için Taktik 3D Görselleştirme Motoru (Panda3D)
# ÖZELLİKLER: Orbital Kamera, Sonsuv Grid, Koni Modeller, Yön Vektörleri
# ==============================================================================
from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties, LineSegs, NodePath, LVector3, TextNode
from direct.task import Task
import numpy as np
import sys

class DogfightViewer(ShowBase):
    def __init__(self):
        # 1. Panda3D'yi kısıtlı modda başlat (Eğitim ortamına tam hakimiyet için)
        super().__init__(windowType='onscreen')
        
        # 2. Pencere Ayarları
        props = WindowProperties()
        props.setTitle("Taktik İt Dalaşı İzleyici - Panda3D")
        props.setSize(1280, 1024)
        self.win.requestProperties(props)
        self.setBackgroundColor(0.1, 0.1, 0.1, 1.0) # Koyu gri tema
        
        # 3. Kendi Kamera Kontrolümüzü Kuruyoruz (Varsayılanı devredışı bırak)
        self.disableMouse()
        self.setup_camera_controls()
        
        # 4. 3D Grid ve Sonsuz Çizgileri Çiz
        self.create_grid(size=50000, spacing=2000)
        
        # 5. Taktik Modelleri ve Vektörleri Yükle
        # Kutular yerine yönü belli olan koni modelleri kullanıyoruz
        self.agent_viz = {}
        self.setup_agents()

        print("Aga: Taktik Viewer hazır. Kamerayı kontrol etmek için:")
        print("- Orta Fare: Döndürme\n- Shift+Orta Fare: Kaydırma\n- Fare Tekerleği: Yakınlaşma")

    # ==========================================================================
    # 3D DÜNYA VE MODELLERİN KURULUMU
    # ==========================================================================
    def create_grid(self, size, spacing):
        """Mekansal algıyı artırmak için 3D Grid ve Eksen Çizgileri çizer."""
        segs = LineSegs()
        
        # Grid çizgileri (Gri)
        segs.setColor(0.3, 0.3, 0.3, 1.0)
        num_lines = int(size / spacing)
        
        for i in range(-num_lines, num_lines + 1):
            offset = i * spacing
            # X eksenine paralel çizgiler
            segs.moveTo(LVector3(-size, offset, 0))
            segs.drawTo(LVector3(size, offset, 0))
            # Y eksenine paralel çizgiler
            segs.moveTo(LVector3(offset, -size, 0))
            segs.drawTo(LVector3(offset, size, 0))
            
        grid_np = NodePath(segs.create())
        grid_np.reparentTo(self.render)
        
        # Ana Eksenler (X: Kırmızı, Y: Yeşil, Z: Mavi)
        axes = LineSegs()
        axes.setThickness(3)
        axes.setColor(0.8, 0.2, 0.2, 1) # X
        axes.moveTo(0,0,0); axes.drawTo(size, 0, 0)
        axes.setColor(0.2, 0.8, 0.2, 1) # Y
        axes.moveTo(0,0,0); axes.drawTo(0, size, 0)
        axes.setColor(0.2, 0.2, 0.8, 1) # Z
        axes.moveTo(0,0,0); axes.drawTo(0, 0, size/2)
        
        axes_np = NodePath(axes.create())
        axes_np.reparentTo(self.render)
    
    def setup_agents(self):
        """Ajan modellerini (gltf) ve Yön Vektörlerini oluşturur."""
        
        # --- AYAR VANALARI ---
        MODEL_SCALE = 5       
        VECTOR_LENGTH = 1500  
        
        # 🔥 MODEL YÖNÜ DÜZELTME (KALİBRASYON) VANASI 🔥
        # Uçağın burnu ok ile aynı yöne bakana kadar bu üçlüyle (H, P, R) oyna:
        # H: Sağa/Sola dönme | P: Aşağı/Yukarı eğilme | R: Sağa/Sola yatma
        MODEL_OFFSET_HPR = (0, 0, 0) # Eğer uçak ters ise (180, 0, 0), dik duruyorsa (0, 90, 0) yap.
        
        for agent_id, color in [("agent_1", (1, 0.2, 0.2, 1)), ("agent_2", (0.2, 0.2, 1, 1))]:
            # Ana Fizik Düğümü (Görünmez Merkez)
            agent_np = self.render.attachNewNode(f"{agent_id}_root")
            
            # 1. Görünen Model (GLTF)
            try:
                model = self.loader.loadModel("/home/ayganyavuz/Desktop/dogfighting_rl/playground/three-d-models/scene.gltf")
            except:
                model = self.loader.loadModel("models/box")

            model.reparentTo(agent_np)
            model.setScale(MODEL_SCALE) 
            model.setColorScale(color[0], color[1], color[2], 1.0) # Kamuflajı bozmadan renklendir
            
            # Modelin kendi eksen bozukluğunu buradan düzeltiyoruz
            model.setHpr(*MODEL_OFFSET_HPR) 
            
            # 2. Gerçek Uçuş Vektörü (Ok)
            vec_np = agent_np.attachNewNode(f"{agent_id}_vector")
            vec_segs = LineSegs()
            vec_segs.setThickness(4)
            vec_segs.setColor(color[0], color[1], color[2], 1)
            
            # Panda3D'de İLERİ YÖN +Y eksenidir. Oku tam Y ekseni (ileri) boyunca çiziyoruz.
            vec_segs.moveTo(0, 0, 0)
            vec_segs.drawTo(0, VECTOR_LENGTH, 0) 
            
            vec_mesh = vec_np.attachNewNode(vec_segs.create())
            
            self.agent_viz[agent_id] = {
                'root': agent_np, # Uçağın ve okun bağlı olduğu ana gövde
                'vector': vec_np
            }
    # ==========================================================================
    # ORBITAL KAMERA KONTROL SİSTEMİ (FARE)
    # ==========================================================================
    def setup_camera_controls(self):
        """Fare olaylarını ve kamera güncelleme görevini başlatır."""
        self.cam_dist = 10000
        self.cam_h = 45
        self.cam_p = -20
        self.cam_target = LVector3(0, 0, 10000) # Bakış merkezi
        
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.dragging = False
        self.panning = False

        # Fare Olaylarını Dinle
        self.accept('mouse3', self.set_dragging, [True])   # Sağ tıkla döndürme
        self.accept('mouse3-up', self.set_dragging, [False])
        
        self.accept('shift-mouse3', self.set_panning, [True]) # Shift+Sağ tıkla kaydırma
        self.accept('shift-mouse3-up', self.set_panning, [False])
        
        self.accept('wheel_up', self.adjust_zoom, [-0.1])   # Yakınlaşma
        self.accept('wheel_down', self.adjust_zoom, [0.1]) # Uzaklaşma

        self.taskMgr.add(self.update_camera_task, "update_camera_task")

    def set_dragging(self, val): self.dragging = val
    def set_panning(self, val): self.panning = val
    def adjust_zoom(self, delta): self.cam_dist = np.clip(self.cam_dist * (1 + delta), 2000, 100000)

    def update_camera_task(self, task):
        """Fare hareketlerine göre kameranın matrisini günceller."""
        if not self.mouseWatcherNode.hasMouse():
            return Task.cont
            
        m_x = self.mouseWatcherNode.getMouseX()
        m_y = self.mouseWatcherNode.getMouseY()
        
        dx = m_x - self.last_mouse_x
        dy = m_y - self.last_mouse_y
        
        # 1. Döndürme (Orbit)
        if self.dragging and not self.panning:
            self.cam_h -= dx * 100
            self.cam_p = np.clip(self.cam_p + dy * 50, -85, 85)
            
        # 2. Kaydırma (Pan)
        elif self.panning:
            self.cam_target.setX(self.cam_target.getX() - dx * self.cam_dist * 0.5)
            self.cam_target.setY(self.cam_target.getY() - dy * self.cam_dist * 0.5)
            
        self.last_mouse_x = m_x
        self.last_mouse_y = m_y
        
        # Kamerayı küresel koordinatlara göre yerleştir
        p_rad = np.radians(self.cam_p)
        h_rad = np.radians(self.cam_h)
        
        c_x = self.cam_target.getX() + self.cam_dist * np.cos(p_rad) * np.sin(h_rad)
        c_y = self.cam_target.getY() - self.cam_dist * np.cos(p_rad) * np.cos(h_rad)
        c_z = self.cam_target.getZ() + self.cam_dist * np.sin(p_rad)
        
        self.cam.setPos(c_x, c_y, c_z)
        self.cam.lookAt(self.cam_target)
        
        return Task.cont

    # ==========================================================================
    # GÜNCELLEME DÖNGÜSÜ (CALLER'DAN GELEN VERİ)
    # ==========================================================================
    def update_world(self, p1_state, p2_state):
        """
        Her step() adımında JSBSim verilerini 3D dünyaya basar.
        p_state = {'x', 'y', 'z', 'roll', 'pitch', 'yaw'}
        """
        states = {"agent_1": p1_state, "agent_2": p2_state}
        
        for agent_id, state in states.items():
            viz = self.agent_viz[agent_id]
            
            # 1. Ana Düğümü Taşı ve Döndür
            viz['root'].setPos(state['x'], state['y'], state['z'])
            
            # Panda3D'nin Yaw açısını JSBSim'in Compass Yaw'ına uyduruyoruz
            viz['root'].setH(state['yaw'] - 180) 
            viz['root'].setP(state['pitch'])
            viz['root'].setR(state['roll'])
            
            # 🔥 O hatalı setYScale satırını tamamen SİLDİK! 
            # Ok zaten uçağın burnuna sabitli, uçakla beraber kendi kendine dönecek.
            
        # 3D motoru manuel olarak 1 kare (frame) renderla!
        self.taskMgr.step()