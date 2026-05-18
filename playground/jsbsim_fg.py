import jsbsim
import time
import math


fdm = jsbsim.FGFDMExec(None)

fdm.set_output_directive('/home/ayganyavuz/Desktop/dogfighting_rl/playground/fg_output.xml')
fdm.load_model('f15') 
# 1. Başlangıç Koşulları (Hız 350'den 180 knot'a düşürüldü)
fdm['ic/h-sl-ft'] = 6000 
fdm['ic/vc-kts'] = 180            # <--- HIZI DÜŞÜRDÜK (Güvenli kapama hızı)
fdm['ic/gamma-deg'] = 0           

# 2. Motorları çalıştır
fdm['propulsion/engine[0]/set-running'] = 1
fdm['propulsion/engine[1]/set-running'] = 1
fdm.run_ic()

fdm['simulation/do_simple_trim'] = 1  # Trim uçağı dengeler ama dişlileri açabilir

# --- İŞTE BURADA INITIAL DEĞERLERİ EL YAPIMI OLARAK SIFIRLIYORUZ ---
fdm['fcs/gear-cmd-norm'] = 0.0       # İniş takımı kolunu kapat
fdm['gear/unit[0]/pos-norm'] = 0.0   # Burun tekerleğini fiziksel olarak kapat
fdm['gear/unit[1]/pos-norm'] = 0.0   # Sol tekerleği fiziksel olarak kapat
fdm['gear/unit[2]/pos-norm'] = 0.0   # Sağ tekerleği fiziksel olarak kapat
# ------------------------------------------------------------------

# Çıktı dosyasını yükle
fdm.set_output_directive('/home/ayganyavuz/Desktop/dogfighting_rl/playground/fg_output.xml')

print("F-15 Uçuşu Başladı! İniş takımları direkt kapalı konumda.")

# Ana döngüye girdiğinizde artık 0. saniyeden itibaren tekerlekler tamamen içeride olacaktır.

# 4. BAĞLANTI (Kesin Dosya Yolu ile)

print("F-15 Uçuşu Başladı! Lütfen FlightGear ekranına geçin...")

dt = fdm.get_delta_t()
next_print = 1.0
# 5. 60 Saniyelik Gerçek Zamanlı Uçuş Döngüsü
print("F-15 Uçuşu Başladı! İniş takımları kapatılıyor...")

while fdm.get_sim_time() <= 60.0:
    
    # --- YENİ: İniş Takımlarını Kapatma Komutu ---
    # Simülasyonun 2. saniyesinde iniş takımlarını kapatma komutu veriyoruz (0.0)
    fdm['fcs/gear-cmd-norm'] = 0.0
        
    fdm.run()
    time.sleep(dt)
    
    # Python konsoluna bilgi yazdır
    if fdm.get_sim_time() >= next_print:
        alt = fdm['position/h-sl-ft']
        mach = fdm['velocities/mach']
        
        # İniş takımının o anki fiziksel konumunu okuyoruz (0 = Kapalı, 1 = Açık)
        # Sistem hidrolik olduğu için 1'den 0'a gelmesi birkaç saniye sürecektir
        gear_status = fdm['gear/gear-pos-norm'] 
        fdm["fcs/elevator-cmd-norm"] = -0.4
        fdm["fcs/throttle-cmd-norm[0]"] = 1.0
        fdm["fcs/throttle-cmd-norm[1]"] = 1.0
        fdm["fcs/"]
        
        print(f"Zaman: {fdm.get_sim_time():.0f}s | İrtifa: {alt:.0f} ft | Mach: {mach:.2f} | Dişli Durumu: {gear_status:.2f}")
        next_print += 1.0
        
print("Simülasyon Bitti.")