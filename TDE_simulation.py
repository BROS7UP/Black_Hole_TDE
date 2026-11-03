# /mnt/c/taichi_win/venv/Scripts/python.exe TDE_simulation_2.py

import math
import taichi as ti
import time

ti.init(arch=ti.gpu, default_fp=ti.f32)



G = 1
M_BH = 10000
R_S = 0.5
R_ISCO = 3.0 * R_S  # Bán kính quỹ đạo tròn bền deepest ones
EPSILON = 1e-2      # Hệ số làm mềm

N = 10000           # Hạt thử nghiệm
DT = 2.0e-4         # Bước thời gian
SUBSTEPS = 20       # Số bước/60fps



pos = ti.Vector.field(3, dtype=ti.f32, shape=N)
vel = ti.Vector.field(3, dtype=ti.f32, shape=N)
mass = ti.field(dtype=ti.f32, shape=N)

# pos for fixed black hole at 0, 0, 0
bh_pos = ti.Vector.field(3, dtype=ti.f32, shape=1)


# KERNEL

@ti.func
def compute_bh_acceleration(p: ti.math.vec3) -> ti.math.vec3:
    """
    Tính gia tốc của black hole
    a_BH = - (G * M_BH / ((r - R_s)^2 + eps^2)) * r_hat
    """
    r_vec = p
    r = r_vec.norm()
    acc = ti.Vector([0.0, 0.0, 0.0])

    # Tính lực khi hạt r > R_S
    if r > R_S:
        r_hat = r_vec / r
        denom = (r - R_S) * (r - R_S) + EPSILON * EPSILON
        acc_mag = (G * M_BH) / denom
        acc = -acc_mag * r_hat

    return acc


@ti.kernel
def init_simulation():
    # Đặt Hố đen cố định tại gốc tọa độ (0, 0, 0)
    bh_pos[0] = ti.Vector([0.0, 0.0, 0.0])

    # Đĩa bồi tu
    for i in range(N):
        # Phân bố
        u = ti.random(dtype=ti.f32)
        r = 2.0 + 10.0 * ti.sqrt(u)
        theta = ti.random(dtype=ti.f32) * 2.0 * math.pi

        # Độ dày đĩa mỏng theo trục
        y_offset = (ti.random(dtype=ti.f32) - 0.5) * (r / 10.0)

        x = r * ti.cos(theta)
        z = r * ti.sin(theta)
        pos[i] = ti.Vector([x, y_offset, z])

        # Vận tốc quỹ đạo tròn
        v_circ = ti.sqrt(G * M_BH * r) / (r - R_S)



        ecc_factor = 0.92 + 0.12 * ti.random(dtype=ti.f32)
        v_mag = v_circ * ecc_factor

        # Vector tiếp tuyến ngược chiều kim đồng hồ
        vel[i] = ti.Vector([-ti.sin(theta) * v_mag, 0.0, ti.cos(theta) * v_mag])



        mass[i] = 1.0e-5


@ti.kernel
def substep():
    for i in range(N):
        r = pos[i].norm()

        # Hạt rơi qua hố đen sẽ
        if r <= R_S + EPSILON:
            pos[i] = ti.Vector([0.0, 0.0, 0.0])
            vel[i] = ti.Vector([0.0, 0.0, 0.0])
        else:
            acc = compute_bh_acceleration(pos[i])
            vel[i] += acc * DT
            pos[i] += vel[i] * DT






def main():
    init_simulation()

    window = ti.ui.Window("Black Hole Sim", (1280, 720), vsync=False)
    window.fps_limit = 120
    canvas = window.get_canvas()
    canvas.set_background_color((0.02, 0.02, 0.04))
    scene = window.get_scene()



    camera = ti.ui.Camera()
    camera.position(0.0, 10.0, 22.0)
    camera.lookat(0.0, 0.0, 0.0)
    camera.up(0.0, 1.0, 0.0)
    camera.fov(55)


    move_speed = 0.5
    while window.running:
        for _ in range(SUBSTEPS):
            substep()

        active_mouse = ti.ui.LMB if window.is_pressed(ti.ui.LMB) else ti.ui.RMB
        camera.track_user_inputs(window, movement_speed=move_speed, hold_key=active_mouse)
        dy_cam = 0.0
        if window.is_pressed(ti.ui.SPACE):
            dy_cam += move_speed
        if window.is_pressed(ti.ui.CTRL):
            dy_cam -= move_speed

        if dy_cam != 0.0:
            camera.position(
                camera.curr_position[0],
                camera.curr_position[1] + dy_cam,
                camera.curr_position[2],
            )
            camera.lookat(
                camera.curr_lookat[0],
                camera.curr_lookat[1] + dy_cam,
                camera.curr_lookat[2],
            )

        scene.set_camera(camera)

        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.0, 15.0, 0.0), color=(1.0, 1.0, 1.0))

        # Render Black Hole
        scene.particles(bh_pos, radius=R_S, color=(0.95, 0.25, 0.05))

        # Render hạt
        scene.particles(pos, radius=0.04, color=(0.35, 0.75, 1.0))


#



#
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
