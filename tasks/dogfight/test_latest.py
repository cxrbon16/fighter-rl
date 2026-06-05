import os
import time
from pathlib import Path
import supersuit as ss
from stable_baselines3 import PPO
import numpy as np
import rerun as rr
from tasks.dogfight.dogfight import SelfPlayDogfightEnv
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[2]


def latest_checkpoint():
    """Highest-step checkpoint from the most recent run subdirectory."""
    checkpoint_base = REPO_ROOT / "tasks/dogfight/models_checkpoints"
    run_dirs = sorted(checkpoint_base.glob("2*"), reverse=True)
    for run_dir in run_dirs:
        ckpts = list(run_dir.glob("ppo_dogfight_*_steps.zip"))
        if ckpts:
            return str(max(ckpts, key=lambda p: int(p.stem.split("_")[-2])))
    return None


def test_dogfight():
    print("Egitilmis Dogfight modeli test ediliyor...")

    # 1. Ortami baslat (standart self-play modeli -> stack_size=4)
    base_env = SelfPlayDogfightEnv(render_mode="debug")
    env = ss.black_death_v3(base_env)
    env = ss.frame_stack_v1(env, stack_size=4)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    model_path = latest_checkpoint()
    try:
        model = PPO.load(model_path)
        print(f"Model basariyla yuklendi: {model_path}")
    except Exception as e:
        print(f"Model bulunamadi ({e}), uclaklar rastgele hareket edecek (Dummy Test).")
        model = None

    # 2. Rerun baslatma
    rr.init("F16_Dogfight_Arena", spawn=True)

    def get_aircraft_quat(roll, pitch, yaw):
        """JSBSim (NED) Euler acilarindan Rerun (ENU) Quaternion'a temiz donusum."""
        base_rot = R.from_euler('x', 90, degrees=True)
        flight_rot = R.from_euler('xyz', [roll, -pitch, 90 - yaw], degrees=True)
        return (flight_rot * base_rot).as_quat()

    rr.log("radar/agent_0/model", rr.Asset3D(path=str(REPO_ROOT / "static/f16.glb")))
    rr.log("radar/agent_1/model", rr.Asset3D(path=str(REPO_ROOT / "static/f16.glb")))

    obs = env.reset()

    trajectories = {'agent_0': [], 'agent_1': []}
    origin_lat = None
    origin_lon = None

    print("Simulasyon basladi! Izlemek icin Rerun motoruna gecin...")

    for step in range(10000):
        if model:
            action, _states = model.predict(obs, deterministic=True)
        else:
            action = np.array([env.action_space.sample() for _ in range(env.num_envs)])

        obs, rewards, dones, infos = env.step(action)

        if any(dones):
            print(f"bitti! Adim: {step} | Ortam sifirlaniyor...")
            trajectories['agent_0'].clear()
            trajectories['agent_1'].clear()
            obs = env.reset()
            origin_lat = None
            continue

        info_0, info_1 = infos[0], infos[1]

        if origin_lat is None:
            origin_lat = info_0['lat']
            origin_lon = info_0['lon']

        lat_to_m = 111320.0
        lon_to_m = 111320.0 * np.cos(np.radians(origin_lat))

        # --- AJAN 0 (MAVI TAKIM) ---
        x_0 = (info_0['lon'] - origin_lon) * lon_to_m
        y_0 = (info_0['lat'] - origin_lat) * lat_to_m
        z_0 = info_0['alt_m']
        rot_0 = get_aircraft_quat(info_0['roll_deg'], info_0['pitch_deg'], info_0['yaw_deg'])
        rr.log("radar/agent_0/model", rr.Transform3D(translation=[x_0, y_0, z_0],
                                                      rotation=rr.Quaternion(xyzw=rot_0), scale=1.0))
        trajectories['agent_0'].append([x_0, y_0, z_0])
        if len(trajectories['agent_0']) > 2:
            rr.log("radar/agent_0/rota", rr.LineStrips3D([trajectories['agent_0']], colors=[100, 150, 255]))

        # --- AJAN 1 (KIRMIZI TAKIM) ---
        x_1 = (info_1['lon'] - origin_lon) * lon_to_m
        y_1 = (info_1['lat'] - origin_lat) * lat_to_m
        z_1 = info_1['alt_m']
        rot_1 = get_aircraft_quat(info_1['roll_deg'], info_1['pitch_deg'], info_1['yaw_deg'])
        rr.log("radar/agent_1/model", rr.Transform3D(translation=[x_1, y_1, z_1],
                                                      rotation=rr.Quaternion(xyzw=rot_1), scale=1.0))
        trajectories['agent_1'].append([x_1, y_1, z_1])
        if len(trajectories['agent_1']) > 2:
            rr.log("radar/agent_1/rota", rr.LineStrips3D([trajectories['agent_1']], colors=[255, 100, 100]))

        # 4. Telemetri (her 5 adimda bir)
        if step % 5 == 0:
            mesafe_m = np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2 + (z_0 - z_1)**2)
            odul_0 = float(rewards[0])
            a0 = action[0]
            a1 = action[1]
            log_metni = (f"Adim: {step:04d} | Mesafe: {mesafe_m:05.0f}m | "
                         f"A0: {info_0['airspeed_kts']:03.0f}kts, {z_0:05.0f}m, Act: [{a0[1]:+.2f}, {a0[0]:+.2f}, {a0[2]:+.2f}] | "
                         f"A1: {info_1['airspeed_kts']:03.0f}kts, {z_1:05.0f}m, Act: [{a1[1]:+.2f}, {a1[0]:+.2f}, {a1[2]:+.2f}] | Odul: {odul_0:+05.2f}")
            print(log_metni)
            rr.log("telemetri/metin", rr.TextLog(log_metni, level=rr.TextLogLevel.INFO))
            rr.log("telemetri/dogfight/aradaki_mesafe_m", rr.Scalars(mesafe_m))
            rr.log("telemetri/dogfight/irtifa_A0", rr.Scalars(z_0))
            rr.log("telemetri/dogfight/irtifa_A1", rr.Scalars(z_1))
            rr.log("telemetri/dogfight/odul_A0", rr.Scalars(odul_0))

        time.sleep(4.0 / 120.0)


if __name__ == "__main__":
    test_dogfight()
