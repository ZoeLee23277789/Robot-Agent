#!/usr/bin/env python
"""把模擬裡 EP 的手臂換成 NexArm（Hiwonder 六軸手臂），存成另一個場景檔，原本的場景不動。可重複執行。

為什麼（2026-11-09）：實體 EP 原本的 2 軸手臂已經沒有了，只剩 NexArm，要裝到 EP 上取代它。模擬裡的手臂不換，
agent 在模擬裡練的動作跟實體對不上（可達範圍差很多：舊手臂前伸 60–220 mm、NexArm 半徑 80–460 mm、高度到 480 mm）。

幾何：官方沒有公開 3D 模型，用實體手臂擬合出來的尺寸（Test_Robot/nexarm_workspace.py 的 FK，11 組實測、最大誤差 0.8 mm）：
    底座旋轉軸 → 肩軸在安裝面上方 110.35 mm → 上臂 227.45 mm → 前臂 145.31 mm → 腕軸到夾爪中心（TCP）129.79 mm
外觀預設用 Hiwonder 官方 ROS 2 倉庫的 URDF 網格（github.com/Hiwonder/NexArm，src/simulations/nexarm_description，STL 走 Git LFS），
下載到 ~/下載/nexarm_description/ 後用 --urdf 指定；官方 URDF 的關節位置跟擬合值差 < 1 mm（肩軸 110.4、上臂 228、前臂 145）。
做法是保留這裡驗證過的運動學骨架，把網格一塊塊掛到對應關節上（先把兩邊擺成 URDF 的零姿勢再掛，位置自動對齊），
自己畫的方塊只當後備（找不到 URDF 時）。

模型結構（全部關節都是運動學模式、連桿不參與物理：伺服器直接設角度，不靠物理去推，見 Test_Robot/nexarm_sim.py）：
    RoboMaster (root)
      └ nexarm_mount（dummy，安裝面：z 朝上、x 朝車頭）
          ├ nexarm_base_shape（底座外殼）
          └ nexarm_j1（底座旋轉，軸 z）
              └ nexarm_turret
                  └ nexarm_j2（肩）→ nexarm_link1 → nexarm_j3（肘）→ nexarm_link2 → nexarm_j4（腕俯仰）→ nexarm_link3
                      └ nexarm_j5（腕旋轉，軸 = 夾爪方向）→ nexarm_gripper（dummy，x 朝夾爪前方、y 朝上、z 側向）
                          ├ nexarm_palm、nexarm_finger_l/r（掛在 nexarm_fj_l/r 這兩個側向滑動關節上）
                          ├ nexarm_tcp（dummy，夾爪中心 = 實體回報的 x/y/z 那一點）
                          └ Vision_sensor（EP 原廠前鏡頭，掛在夾爪下緣、跟著 roll 轉，見 mount_ep_camera）
肩、肘、腕俯仰三個關節的 z 軸都是側向（手臂平面的法線），角度 0 = 連桿水平朝前、正值 = 往上抬。

原本的 RoboMaster 手臂不刪（之前場景裡被拆過的模型外掛拒絕驅動），整組設成看不見、不碰撞、相機拍不到。

    .venv/bin/python Env/add_nexarm.py                       # 讀目前載入的場景，加 NexArm，存成 Capstone_Playground_nexarm.ttt
    .venv/bin/python Env/add_nexarm.py --forward 0.08 --height 0.125   # 安裝位置（離車身中心往前幾公尺、安裝面離地高度）
    .venv/bin/python Env/add_nexarm.py --dry-run             # 只改執行中的場景，不存檔
之後用新場景：COPPELIASIM_SCENE=scenes/Capstone_Playground_nexarm.ttt ./start_sim_headless.sh
"""

import argparse
import math
import os
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

SIM_DIR = os.path.expanduser("~/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04")
OUT_SCENE = os.path.join(SIM_DIR, "scenes/Capstone_Playground_nexarm.ttt")

# 擬合出來的運動學（m）：跟 Test_Robot/nexarm_workspace.py 的 FK 同一組
AX, AZ = -0.00012, 0.11035
L1, L2, L3 = 0.22745, 0.14531, 0.12979
# 官方圖面（rcdrone.top 產品頁，2026-11-09 使用者提供）：底座 111×111、外殼高 69、底座到肩軸 110、連桿 225/145/130、
# 夾爪外寬 51 mm、底座旋轉 270°、肩/肘/腕 240°/270°/200°、整支 1.3 kg。跟擬合值最大差 2.5 mm（上臂，含肩軸座偏移）。
BASE_SIDE, J1_H = 0.111, 0.069
GRIPPER_WIDTH = 0.051            # 兩指外緣全開時的寬度
FINGER = (0.045, 0.022, 0.008)   # 手指：沿夾爪方向長、上下寬、側向厚
GAP_OPEN = GRIPPER_WIDTH - 2 * FINGER[2]   # 全開時兩指內緣距離 35 mm
MASS_KG = 1.3
JOINT_RANGE_DEG = {"j1": 270}   # 只用底座的；其他見 build() 的註解

BLACK, GREY, ORANGE = [0.12, 0.12, 0.12], [0.35, 0.35, 0.38], [0.95, 0.45, 0.10]
LABEL, PAD = [0.92, 0.92, 0.92], [0.55, 0.55, 0.58]   # 伺服標籤、矽膠指墊


def find_robot(sim):
    roots = [o for o in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 2) if sim.getObjectAlias(o) == "RoboMaster"]
    good = [h for h in roots if any(sim.getObjectAlias(o) == "GyroSensor" for o in sim.getObjectsInTree(h, sim.handle_all, 0))]
    if not roots:
        raise SystemExit("找不到 /RoboMaster")
    return (good or roots)[0]


def child(sim, root, alias):
    return sim.getObject("./" + alias, {"proxy": root, "noError": True})


def heading(sim, root):
    """車頭方向（世界 yaw，弧度）＝ 前輪中點 − 後輪中點，跟 robot_server 用同一套（相機會跟手臂動，輪子不會）。"""
    pos = {}
    for o in sim.getObjectsInTree(root, sim.object_joint_type, 0):
        a = sim.getObjectAlias(o)
        if a in ("front_left_wheel_joint", "front_right_wheel_joint", "rear_left_wheel_joint", "rear_right_wheel_joint"):
            pos[a] = sim.getObjectPosition(o, sim.handle_world)
    fx = (pos["front_left_wheel_joint"][0] + pos["front_right_wheel_joint"][0]) / 2
    fy = (pos["front_left_wheel_joint"][1] + pos["front_right_wheel_joint"][1]) / 2
    rx = (pos["rear_left_wheel_joint"][0] + pos["rear_right_wheel_joint"][0]) / 2
    ry = (pos["rear_left_wheel_joint"][1] + pos["rear_right_wheel_joint"][1]) / 2
    return math.atan2(fy - ry, fx - rx)


def visual_shape(sim, kind, size, color):
    h = sim.createPrimitiveShape(kind, list(size), 0)
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)        # 不參與物理：跟著父物件走
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
    # 看得到、相機拍得到、距離感測器偵測得到（實體的 ToF 也會看到手臂）
    sim.setObjectSpecialProperty(h, sim.objectspecialproperty_renderable | sim.objectspecialproperty_collidable
                                 | sim.objectspecialproperty_measurable | sim.objectspecialproperty_detectable)
    return h


def kin_joint(sim, kind, alias, parent, pos, orient, size=(0.02, 0.005)):
    j = sim.createJoint(kind, sim.jointmode_kinematic, 0, list(size))
    sim.setObjectAlias(j, alias)
    sim.setObjectParent(j, parent, False)
    sim.setObjectPosition(j, parent, list(pos))
    sim.setObjectOrientation(j, parent, list(orient))
    return j


def attach(sim, h, alias, parent, pos, orient=(0.0, 0.0, 0.0)):
    sim.setObjectAlias(h, alias)
    sim.setObjectParent(h, parent, False)
    sim.setObjectPosition(h, parent, list(pos))
    sim.setObjectOrientation(h, parent, list(orient))
    return h


def remove_nexarm(sim, root):
    m = child(sim, root, "nexarm_mount")
    if m != -1:
        sim.removeObjects(sim.getObjectsInTree(m, sim.handle_all, 0))
        print("移除舊的 NexArm 模型")


def hide_old_arm(sim, root):
    """原本的 RoboMaster 手臂（含相機、夾爪）：看不見、不碰撞、不被感測器偵測，物理上也不再碰任何東西。
    升降柱留著（前方距離感測器掛在升降座上），但它的方柱也藏起來。"""
    fs = child(sim, root, "arm_base_attachment_joint")
    if fs == -1:
        print("找不到 arm_base_attachment_joint，跳過隱藏舊手臂")
        return 0
    n = 0
    targets = sim.getObjectsInTree(fs, sim.handle_all, 0)
    rod = child(sim, root, "arm_lift_rod")
    if rod != -1:
        targets.append(rod)
    for h in targets:
        sim.setObjectSpecialProperty(h, 0)
        try:
            sim.setObjectInt32Param(h, sim.objintparam_visibility_layer, 0)
        except Exception:
            pass
        if sim.getObjectType(h) == sim.object_shape_type:
            sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
        n += 1
    print("舊手臂 %d 個物件已隱藏（看不見、不碰撞、相機拍不到）" % n)
    return n


def box(sim, parent, alias, size, pos, color, orient=(0.0, 0.0, 0.0)):
    return attach(sim, visual_shape(sim, sim.primitiveshape_cuboid, size, color), alias, parent, pos, orient)


def cyl(sim, parent, alias, d, h, pos, color, orient=(0.0, 0.0, 0.0)):
    return attach(sim, visual_shape(sim, sim.primitiveshape_cylinder, (d, d, h), color), alias, parent, pos, orient)


def servo(sim, parent, alias, pos, size=(0.040, 0.054, 0.045), orient=(0.0, 0.0, 0.0)):
    """匯流排伺服外觀：黑盒子 + 側面一條白標籤（實體 HX-30HM/65HM 側面有白字標籤）。"""
    b = box(sim, parent, alias, size, pos, BLACK, orient)
    box(sim, b, alias + "_label", (size[0] * 0.6, 0.0005, size[2] * 0.45), (0.0, size[1] / 2 + 0.0002, 0.0), LABEL)
    return b


# 連桿外觀：兩根平行的 15 mm 鋁擠型（官方正視圖：上臂、前臂都是雙軌），軌距依實體照片估
RAIL, RAIL_GAP_UPPER, RAIL_GAP_FORE = 0.015, 0.040, 0.034


def rails(sim, parent, alias, length, gap, pos_x0):
    """沿父座標 +x 方向、從 pos_x0 開始長 length 的兩根軌，側向（z）分開 gap。"""
    for side, sgn in (("a", 1.0), ("b", -1.0)):
        box(sim, parent, "%s_%s" % (alias, side), (length, RAIL, RAIL), (pos_x0 + length / 2, 0.0, sgn * gap / 2), BLACK)


def build(sim, root, forward, height):
    yaw = heading(sim, root)
    rp = sim.getObjectPosition(root, sim.handle_world)

    mount = sim.createDummy(0.02)
    sim.setObjectAlias(mount, "nexarm_mount")
    car = child(sim, root, "arm_lift_carriage")
    if car != -1 and forward is None:
        # 裝在原本手臂的升降座（Env/add_arm_lift.py）上：底座坐在升降座頂面、車身正中，lift_arm 就能把整支手臂升高
        cp = sim.getObjectPosition(car, sim.handle_world)
        top = cp[2] + sim.getObjectFloatParam(car, sim.objfloatparam_objbbox_max_x)   # 升降座的 local x 朝上
        mx, my, height = cp[0], cp[1], top
        parent = car
        print("底座掛在升降座 arm_lift_carriage 上（頂面離地 %.3f m）" % top)
    else:
        if forward is None:
            forward = 0.08
            mx, my = rp[0] + forward * math.cos(yaw), rp[1] + forward * math.sin(yaw)
        parent = root
    sim.setObjectPosition(mount, sim.handle_world, [mx, my, height])
    sim.setObjectOrientation(mount, sim.handle_world, [0.0, 0.0, yaw])
    sim.setObjectParent(mount, parent, True)

    # 底座：111×111×69 黑盒 + 頂面轉盤
    box(sim, mount, "nexarm_base_shape", (BASE_SIDE, BASE_SIDE, J1_H), (0.0, 0.0, J1_H / 2), BLACK)
    cyl(sim, mount, "nexarm_base_ring", 0.100, 0.006, (0.0, 0.0, J1_H + 0.003), GREY)

    j1 = kin_joint(sim, sim.joint_revolute_subtype, "nexarm_j1", mount, (0.0, 0.0, J1_H + 0.006), (0.0, 0.0, 0.0))
    sim.setJointInterval(j1, False, [-math.radians(JOINT_RANGE_DEG["j1"] / 2), math.radians(JOINT_RANGE_DEG["j1"])])
    # 轉台：底板 + 65 kg·cm 大伺服（立著）+ 兩片 U 型支架，肩軸在安裝面上方 AZ
    turret_h = AZ - (J1_H + 0.006)
    turret = box(sim, j1, "nexarm_turret", (0.090, 0.090, 0.006), (0.0, 0.0, 0.003), GREY)
    servo(sim, turret, "nexarm_servo_shoulder", (0.0, 0.0, turret_h / 2), size=(0.044, 0.060, turret_h - 0.012))
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        box(sim, turret, "nexarm_bracket_" + side, (0.050, 0.004, turret_h + 0.01), (0.0, sgn * 0.036, turret_h / 2 + 0.002), BLACK)

    # 肩：繞 x 轉 +90° → 關節 z 軸 = 父的 −y（側向），關節 x = 前、y = 上；正角度把連桿往上抬
    # turret 這塊板本身在 j1 上方 3 mm，肩軸相對它要扣掉，世界高度才剛好是 AZ
    j2 = kin_joint(sim, sim.joint_revolute_subtype, "nexarm_j2", turret, (AX, 0.0, turret_h - 0.003), (math.pi / 2, 0.0, 0.0))
    link1 = box(sim, j2, "nexarm_link1", (0.050, 0.046, 0.072), (0.0, 0.0, 0.0), BLACK)       # 肩軸座（雙軌夾住伺服）
    rails(sim, link1, "nexarm_upper_rail", L1 - 0.05, RAIL_GAP_UPPER, 0.025)
    j3 = kin_joint(sim, sim.joint_revolute_subtype, "nexarm_j3", j2, (L1, 0.0, 0.0), (0.0, 0.0, 0.0))
    sim.setObjectParent(j3, link1, True)
    # 肘：伺服橫放在雙軌之間 + 側面三角形板（用薄方板代替）
    link2 = servo(sim, j3, "nexarm_link2", (0.0, 0.0, 0.0), size=(0.046, 0.040, 0.034))
    box(sim, link2, "nexarm_elbow_plate", (0.070, 0.050, 0.003), (0.015, 0.0, RAIL_GAP_FORE / 2 + 0.0025), GREY)
    rails(sim, link2, "nexarm_fore_rail", L2 - 0.045, RAIL_GAP_FORE, 0.023)
    j4 = kin_joint(sim, sim.joint_revolute_subtype, "nexarm_j4", j3, (L2, 0.0, 0.0), (0.0, 0.0, 0.0))
    sim.setObjectParent(j4, link2, True)
    # 腕：俯仰伺服 + 旋轉小伺服（HX-12H）
    wrist = servo(sim, j4, "nexarm_link3", (0.0, 0.0, 0.0), size=(0.046, 0.040, 0.034))
    servo(sim, wrist, "nexarm_servo_roll", (0.040, 0.0, 0.0), size=(0.030, 0.030, 0.026))
    j5 = kin_joint(sim, sim.joint_revolute_subtype, "nexarm_j5", j4, (0.05, 0.0, 0.0), (0.0, math.pi / 2, 0.0))
    sim.setObjectParent(j5, wrist, True)
    grip = sim.createDummy(0.01)
    sim.setObjectAlias(grip, "nexarm_gripper")
    sim.setObjectParent(grip, j5, False)
    sim.setObjectPosition(grip, j4, [0.05, 0.0, 0.0])
    sim.setObjectOrientation(grip, j4, [0.0, 0.0, 0.0])     # 跟腕俯仰關節同方向：x 朝前、y 朝上、z 側向

    # 平行夾爪：導軌座 51 mm 寬（官方）、兩片三角形指墊（薄方板 + 斜面用兩塊疊出）
    finger_x = L3 - 0.05            # 手指中心 = TCP（grip 在腕軸前 0.05 m）
    box(sim, grip, "nexarm_palm", (0.030, 0.034, GRIPPER_WIDTH), (0.015, 0.0, 0.0), BLACK)
    box(sim, grip, "nexarm_rail_housing", (finger_x - 0.030, 0.012, GRIPPER_WIDTH), (0.030 + (finger_x - 0.030) / 2, -0.011, 0.0), GREY)
    half = GAP_OPEN / 2 + FINGER[2] / 2
    for side, sign in (("l", 1.0), ("r", -1.0)):
        fj = kin_joint(sim, sim.joint_prismatic_subtype, "nexarm_fj_" + side, grip, (finger_x, 0.0, 0.0),
                       (0.0, 0.0, 0.0) if sign > 0 else (math.pi, 0.0, 0.0), size=(0.01, 0.004))
        sim.setJointInterval(fj, False, [0.0, 0.06])
        sim.setJointPosition(fj, half)
        f = box(sim, fj, "nexarm_finger_" + side, FINGER, (0.0, 0.0, 0.0), BLACK)
        box(sim, f, "nexarm_pad_" + side, (FINGER[0] * 0.8, FINGER[1] * 0.8, 0.002), (0.0, 0.0, -sign * (FINGER[2] / 2 + 0.001) * (1 if side == "l" else -1)), PAD)
    tcp = sim.createDummy(0.008)
    attach(sim, tcp, "nexarm_tcp", grip, (finger_x, 0.0, 0.0))

    # 手腕相機（2026-11-09～11-10）已經不建：實體 EP 的相機修好了（11-11），裝在車頭，模擬也用 EP 原廠的 Vision_sensor，見 mount_ep_camera()
    print("NexArm 模型已建立：安裝面世界座標 (%.3f, %.3f)、離地 %.3f m，質量 %.1f kg 未加進物理（連桿是純視覺）"
          % (mx, my, height, MASS_KG))
    return mount, tcp


def mount_ep_camera(sim, root, gripper, forward_m=0.015, below_m=0.040):
    """EP 原廠前鏡頭（模型裡的 Vision_sensor，1280×720、視角 102°）掛在夾爪總成的下緣、腕部 roll 那一節（nexarm_gripper，
    j5 的子物件，跟著 360° 旋轉）：使用者 2026-11-11「放在機械抓夾的下緣，可以轉 360 度的那個地方」。
    夾爪座標：+x 朝指尖、+y 上、z 側向。相機中心在腕軸前 forward_m、夾爪原點下 below_m（4 cm：殼頂剛好貼到夾爪底面，
    用包圍盒估會低 2 cm 懸空），看出去沿 +x、畫面上方 +y。相機殼網格 camera_link_visual 跟著並顯示。實體量到不同就改參數再 saveScene。"""
    cam = sim.getObject("./Vision_sensor", {"proxy": root})
    mesh = sim.getObject("./camera_link_visual", {"proxy": root, "noError": True})
    sim.setObjectParent(cam, gripper, True)
    sim.setObjectMatrix(cam, gripper, [0, 0, 1, forward_m, 0, 1, 0, -below_m, -1, 0, 0, 0.0])
    sim.setObjectInt32Param(cam, sim.objintparam_visibility_layer, 0)
    sim.setObjectFloatParam(cam, sim.visionfloatparam_near_clipping, 0.02)
    if mesh != -1:
        sim.setObjectParent(mesh, cam, True)
        sim.setObjectMatrix(mesh, cam, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
        sim.setObjectInt32Param(mesh, sim.objintparam_visibility_layer, 1)
    return cam


# 官方 URDF 的零姿勢（手臂直立、前臂折回）對應骨架的角度：從匯入後各關節的世界座標算出來的弦方向
URDF_ZERO = (81.2, -79.1, 0.0)   # p1, p2, pitch
# URDF 的 visual 網格 → 掛到哪個骨架物件；(URDF link 名, 骨架 alias, 新 alias)
URDF_LINKS = [("base_link", "mount", "base"), ("link1", "j1", "link1"), ("link2", "j2", "link2"), ("link3", "j3", "link3"),
              ("link4", "j4", "link4"), ("camera_link", "j4", "camera"), ("link5", "j5", "link5"), ("gripper_link", "j5", "gripper"),
              ("gripper_base_link", "j5", "gripper_base"),
              # URDF 的 left（+y）在骨架是 −z 那側 = fj_r；掛反了墊片會朝外（2026-11-09 使用者抓到）
              ("right_jaw_link", "fj_l", "jaw_l"), ("left_jaw_link", "fj_r", "jaw_r")]


def graft_urdf(client, sim, root, urdf_path):
    """把官方 URDF 的網格掛到骨架上。匯入時 package:// 路徑換成絕對路徑（寫一份暫存 URDF）。"""
    import tempfile
    simURDF = client.require("simURDF")
    # 上一次匯入到一半留下的 URDF 模型（頂層的 base_link_respondable）先清掉
    for o in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 2):
        if sim.getObjectAlias(o) == "base_link_respondable":
            sim.removeModel(o)
            print("移除殘留的 URDF 模型")
    pkg_dir = os.path.dirname(os.path.dirname(urdf_path))
    text = open(urdf_path, encoding="utf-8").read()
    import re
    text = re.sub(r"package://[^/]+/", pkg_dir + "/", text)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False, encoding="utf-8")
    tmp.write(text)
    tmp.close()
    name, handles = getattr(simURDF, "import")(tmp.name, 32)     # bit5：不要自動抬到地面上
    os.unlink(tmp.name)
    urdf_root = handles[0]
    H = lambda a: child(sim, root, "nexarm_" + a)
    j = {n: H(n) for n in ("j1", "j2", "j3", "j4", "j5")}
    p1, p2, pitch = URDF_ZERO
    for jn, v in (("j1", 0.0), ("j2", p1), ("j3", p2 - p1), ("j4", pitch - p2), ("j5", 0.0)):
        sim.setJointPosition(j[jn], math.radians(v))
    sim.setObjectPose(urdf_root, sim.handle_world, sim.getObjectPose(H("mount"), sim.handle_world))
    by_alias = {sim.getObjectAlias(o): o for o in sim.getObjectsInTree(urdf_root, sim.handle_all, 0)}
    n = 0
    for link, target, new in URDF_LINKS:
        mesh = by_alias.get(link + "_visual")
        if mesh is None:
            print("URDF 裡沒有 %s_visual，跳過" % link)
            continue
        sim.setObjectParent(mesh, H(target), True)
        sim.setObjectAlias(mesh, "nexarm_mesh_" + new)
        sim.setObjectInt32Param(mesh, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(mesh, sim.shapeintparam_respondable, 0)
        sim.setObjectSpecialProperty(mesh, sim.objectspecialproperty_renderable | sim.objectspecialproperty_collidable
                                     | sim.objectspecialproperty_measurable | sim.objectspecialproperty_detectable)
        sim.setObjectInt32Param(mesh, sim.objintparam_visibility_layer, 1)
        sim.setShapeColor(mesh, None, sim.colorcomponent_ambient_diffuse, [0.13, 0.13, 0.14])
        n += 1
    # 兩片夾爪網格：URDF 的左右跟骨架的側向軸相反、而且相距 77 mm，掛上去之後要沿側向平移，讓「手指關節 = 半指厚（全合）」時
    # 兩片的內側面剛好在中線相碰（2026-11-09 實測：實體合了、分身沒合，就是沒做這步）
    grip = H("gripper")
    for side, sign in (("l", 1.0), ("r", -1.0)):
        fj, mesh = H("fj_" + side), H("mesh_jaw_" + side)
        q = sim.getJointPosition(fj) * 1000.0
        mm = [sim.getObjectFloatParam(mesh, p) for p in (sim.objfloatparam_objbbox_min_x, sim.objfloatparam_objbbox_min_y, sim.objfloatparam_objbbox_min_z,
                                                          sim.objfloatparam_objbbox_max_x, sim.objfloatparam_objbbox_max_y, sim.objfloatparam_objbbox_max_z)]
        M = sim.getObjectMatrix(mesh, grip)
        zs = [M[8] * x + M[9] * y + M[10] * z + M[11] for x in (mm[0], mm[3]) for y in (mm[1], mm[4]) for z in (mm[2], mm[5])]
        inner = (min(zs) if sign > 0 else max(zs)) * 1000.0
        want = sign * (q - FINGER[2] * 1000.0 / 2)
        p = sim.getObjectPosition(mesh, grip)
        p[2] += (want - inner) / 1000.0
        sim.setObjectPosition(mesh, grip, p)
    # 官方模型的 camera_link 網格（手腕上那顆相機的外觀）不要：實體 NexArm 上沒有相機
    try:
        sim.removeObjects([H("mesh_camera")])
    except Exception:
        pass
    j4 = j["j4"]
    # 方向要拿 j4 的「子物件」（nexarm_link3）的座標系：CoppeliaSim 關節物件自己的座標系不含關節角，只有子物件會跟著轉
    # （2026-11-09 用 j4 自己的 x 當前方，相機差了一個腕角度、變成朝下）
    # 2026-11-10 實測（headless 場景、夾爪水平時量的）：上面那套用 link3 軸算出來的感測器落在外殼網格裡面、而且朝下看自己的底座，
    # 畫面永遠是升降柱和車殼（使用者：「相機是不是裝反了」）。改成直接給「相對 j4」的固定姿態：外殼網格中心在 j4 前 6.45 cm、上 3.75 cm、
    # 沿前方長 ±3.8 cm，感測器放到殼外前緣（j4 前 11 cm），z 朝前、y 朝上（euler (π/2, π/2, −π/2)）。驗證：tcp 在前 11 cm、下 4 cm，
    # 3.7 m 外的 pillar_blue、1.8 m 外的 mat_green 都在前方視野內。
    # 注意：模擬「停止」會把播放中改的物件姿態全部還原，這種修正要在停止狀態下改、再 saveScene 才會留下來。
    sim.removeModel(urdf_root)
    hidden = 0
    for o in sim.getObjectsInTree(H("mount"), sim.object_shape_type, 0):
        if not sim.getObjectAlias(o).startswith("nexarm_mesh_"):
            sim.setObjectSpecialProperty(o, 0)
            sim.setObjectInt32Param(o, sim.objintparam_visibility_layer, 0)
            hidden += 1
    # 回預設姿勢
    for jn, v in (("j2", 119.8), ("j3", -143.2), ("j4", 23.4)):
        sim.setJointPosition(j[jn], math.radians(v))
    print("官方網格已掛上骨架：%d 塊，方塊外觀 %d 塊已隱藏" % (n, hidden))


def verify(sim, root):
    """設幾組關節角，比對 nexarm_tcp 在安裝面座標系裡的位置跟擬合 FK 算的一不一致（不用跑模擬）。"""
    j = {n: child(sim, root, "nexarm_" + n) for n in ("j1", "j2", "j3", "j4", "j5")}
    mount, tcp = child(sim, root, "nexarm_mount"), child(sim, root, "nexarm_tcp")
    worst = 0.0
    for q1, p1, p2, p3 in ((0, 119.8, -23.4, 0.0), (0, 60, 10, -45), (30, 90, -60, -90), (-45, 100, 0, 0)):
        sim.setJointPosition(j["j1"], math.radians(q1))
        sim.setJointPosition(j["j2"], math.radians(p1))
        sim.setJointPosition(j["j3"], math.radians(p2 - p1))
        sim.setJointPosition(j["j4"], math.radians(p3 - p2))
        r = AX + L1 * math.cos(math.radians(p1)) + L2 * math.cos(math.radians(p2)) + L3 * math.cos(math.radians(p3))
        z = AZ + L1 * math.sin(math.radians(p1)) + L2 * math.sin(math.radians(p2)) + L3 * math.sin(math.radians(p3))
        want = (r * math.cos(math.radians(q1)), r * math.sin(math.radians(q1)), z)
        got = sim.getObjectPosition(tcp, mount)
        err = math.dist(want, got)
        worst = max(worst, err)
        print("  q1=%+4.0f p=(%5.1f,%6.1f,%6.1f) → FK (%.1f, %.1f, %.1f) mm，模型 TCP (%.1f, %.1f, %.1f) mm，差 %.2f mm"
              % (q1, p1, p2, p3, *(v * 1000 for v in want), *(v * 1000 for v in got), err * 1000))
    for n in ("j1", "j2", "j3", "j4", "j5"):
        sim.setJointPosition(j[n], 0.0)
    # 回到預設姿勢 (150, 0, 250) 夾爪水平
    sim.setJointPosition(j["j2"], math.radians(119.8))
    sim.setJointPosition(j["j3"], math.radians(-143.2))
    sim.setJointPosition(j["j4"], math.radians(23.4))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", type=float, default=None,
                    help="安裝面中心離車身中心往車頭幾公尺。不給（預設）就裝在升降座 arm_lift_carriage 上、車身正中；給了就直接掛在底盤上")
    ap.add_argument("--height", type=float, default=0.125, help="安裝面離地高度（m），預設是 EP 上蓋的高度")
    ap.add_argument("--urdf", default="~/下載/nexarm_description/urdf/nexarm.urdf",
                    help="官方 URDF（網格在同層的 ../meshes/）；給空字串就只用方塊外觀")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--port", type=int, default=23000)
    args = ap.parse_args()

    client = RemoteAPIClient(port=args.port)
    sim = client.require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        while sim.getSimulationState() != sim.simulation_stopped:
            time.sleep(0.1)
        print("模擬已停止")
    root = find_robot(sim)
    remove_nexarm(sim, root)
    hide_old_arm(sim, root)
    build(sim, root, args.forward, args.height)
    worst = verify(sim, root)
    if args.urdf and os.path.exists(os.path.expanduser(args.urdf)):
        graft_urdf(client, sim, root, os.path.expanduser(args.urdf))
    elif args.urdf:
        print("找不到 URDF %s，保留方塊外觀" % args.urdf)
    print("模型 TCP 跟擬合 FK 最大差 %.2f mm" % (worst * 1000))
    if worst > 0.001:
        raise SystemExit("模型跟 FK 對不上，不存檔")
    # EP 原廠前鏡頭掛到夾爪下緣（跟著腕部 roll 轉）
    mount_ep_camera(sim, root, child(sim, root, "nexarm_gripper"))
    if args.dry_run:
        print("--dry-run：沒有存檔")
        return
    sim.saveScene(OUT_SCENE)
    print("已存成新場景：", OUT_SCENE, "（原本的 Capstone_Playground.ttt 沒動）")


if __name__ == "__main__":
    main()
