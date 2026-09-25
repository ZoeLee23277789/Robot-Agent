"""跑 3 秒模擬，回報動態物件的高度變化與地板的物理設定，用來查物件掉穿地板的原因。
用法：python diag_physics.py（模擬停止狀態下執行）"""
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

sim = RemoteAPIClient().require("sim")
names = ["/RoboMaster", "/Playground/small_red", "/Playground/small_yellow", "/Playground/ball_purple",
         "/Playground/foam_cube_blue", "/Playground/pit_balls/pit_ball_0"]
hs = {n: sim.getObject(n, {"noError": True}) for n in names}
before = {n: sim.getObjectPosition(h, -1) for n, h in hs.items() if h != -1}

for path in ["/Playground/ground/ground_slab", "/Playground/floor_tiles/tile_0_0", "/Playground/small_red"]:
    h = sim.getObject(path, {"noError": True})
    if h == -1:
        print(path, "不存在"); continue
    st = sim.getObjectInt32Param(h, sim.shapeintparam_static)
    rp = sim.getObjectInt32Param(h, sim.shapeintparam_respondable)
    mask = sim.getObjectInt32Param(h, sim.shapeintparam_respondable_mask)
    bb = sim.getShapeBB(h) if hasattr(sim, "getShapeBB") else None
    print(f"{path}: static={st} respondable={rp} mask={mask:#06x} pos={[round(v,3) for v in sim.getObjectPosition(h,-1)]} bb={bb}")

print("engine:", sim.getInt32Param(sim.intparam_dynamic_engine), "(0=Bullet 1=ODE 2=Vortex 3=Newton 4=MuJoCo)")
sim.startSimulation()
time.sleep(3.0)
after = {n: sim.getObjectPosition(h, -1) for n, h in hs.items() if h != -1}
sim.stopSimulation()
for n in before:
    b, a = before[n], after[n]
    print(f"{n:38s} z: {b[2]:+.3f} -> {a[2]:+.3f}   xy: ({b[0]:+.2f},{b[1]:+.2f}) -> ({a[0]:+.2f},{a[1]:+.2f})")
