# 代码解读
参考：
[读《Visual Whole-Body for Loco-Manipulation》①low](https://blog.csdn.net/weixin_40459958/article/details/141609348)

代码调用顺序：
```
b1z1_config.py > maniploco_rewards.py > manip_loco.py > task_registry.py > train.py
```
# 奖励设计：
1. 手臂Z1奖励：
   tracking_ee_world = 0.8（手臂末端位置xyz奖励）
2. 四足B1奖励：
   #-------Gait control rewards ---------
    tracking_contacts_shaped_force = -2.0 # Only works when `observing_gait_commands` is true
    tracking_contacts_shaped_vel = -2.0 # Only works when `observing_gait_commands` is true
    feet_air_time = 2.0
    feet_height = 1.0
    #-------Tracking rewards ----------
    tracking_lin_vel_max = 2.0 
    tracking_ang_vel = 0.5

    delta_torques = -1.0e-7/4.0
    torques = -2.5e-5 
    stand_still = 1.0 
    walking_dof = 1.5
    alive = 1.0
    lin_vel_z = -1.5
    roll = -2

    #common rewards
    ang_vel_xy = -0.2 
    dof_acc = -7.5e-7 
    collision = -10.
    action_rate = -0.015
    dof_pos_limits = -10.0
    delta_torques = -1.0e-7
    hip_pos = -0.3
    work = -0.003
    feet_jerk = -0.0002
    feet_drag = -0.08
    feet_contact_forces = -0.001
    base_height = -5.0


# 手臂Z1
## def step(self, actions):
1. 接收12自由度腿部动作actions
2. 基于手臂末端位姿增量$dpose$计算手臂关节角度arm_pos_targets
   1. 根据目标与实际的末端位姿差计算末端位置增量$dpose$
   2. 调用_control_ik()函数计算末端位姿增量$dpose$对应的关节角度增量$u$
   3. 计算手臂关节角度目标arm_pos_targets
3. 输入手臂关节位置控制+腿部关节力控制，执行动作推进仿真
4. 后处理 self.post_physics_step() 更新环境状态，计算观测值和奖励，重置环境等
   1. gym.refresh_xxx()函数更新环境状态[_init_buffers()函数中的变量也自动同步更新]
   2. 计算base_xx基本物理量
   3. 更新接触状态滤波
   4. _post_physics_step_callback()执行物理步骤之后的通用计算(命令重新采样,更新步态接触目标,随机推机器人)
   5. 更新eef goal: _update_curr_ee_goal()函数
      1. 更新机器人手臂或其他末端执行器的目标位置。这对实现复杂的操控任务至关重要。curr_ee_goal_cart_world在该函数中被更新，其值用于计算dpose
   6. 检查终止条件 check_termination()
      1. 不可预期接触力判断终止条件(terminate_after_contacts_on 未设)
      2. 姿态终止条件(pitch，roll> 0.8)
      3. 高度终止条件
      4. 超时终止条件
   7. 计算compute_reward()奖励
      1. rew_buf += rew*rew_scales，其中rew_scales为奖励权重
      2. 手臂奖励仅跟踪3维末端位置xyz, _reward_tracking_ee_world()
   8. reset_idx() self.reset_buf 中存在非零值时，重置环境
      1. update curriculum
      2. reset robot states
      3. reset buffers
   9.  计算观测 compute_observations()
      1. 包括机器人的姿态、速度、关节状态等信息
      2. 计算手臂基座世界坐标系位置xyz: arm_base_pos
      3. 计算手臂末端相对于基座相对笛卡尔坐标xyz: ee_goal_local_cart
      4. 待机模式（self.stand_by 为 True），命令为0
      5. 构建观测张量（obs_buf）: num_envs,66
         1. 机器人身体的姿态（2维）
         2. 机器人底座的角速度（3维）
         3. 关节位置相对于默认位置的偏差（18维，不包括末端执行器的关节）
         4. 关节速度（18维，不包括夹爪关节）
         5. 最近一次的动作历史（12维，只包括腿部关节）
         6. 足部接触状态（4维）
         7. 线速度命令（3维）
         8. 末端执行器目标位置的局部坐标（3维）
         9. 末端执行器目标的方向（3维，但被设置为零）
5. 返回 self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.arm_rew_buf, self.reset_buf, self.extras



1. _control_ik()函数
A = J_eef * J_eef^T + λ^2 * I;
u = J_eef^T * (J_eef * J_eef^T + λ^2 * I)^-1 * dpose
其中：
$x = A^{-1} * dpose$
$u = J_{eef}^T* x$
其中，u为关节角度增量$\Delta\theta$，λ为惩罚参数，I为单位矩阵，dpose为末端位姿增量$\Delta\P_{ee}$，J_eef为末端Jacobian矩阵。

2. 奖励函数：
   1. #-------Gait control rewards ---------
      tracking_contacts_shaped_force = -2.0 # works when `observing_gait_commands` is true
      tracking_contacts_shaped_vel = -2.0 # works when `observing_gait_commands` is true
      feet_air_time = 2.0     # v
      feet_height = 1.0       # v

      #-------Tracking rewards ----------
      tracking_lin_vel_max = 2.0 # v
      tracking_ang_vel = 0.5  # v

      torques = -2.5e-5   # v
      stand_still = 1.0   # v
      walking_dof = 1.5   # v
      alive = 1.0         # v
      lin_vel_z = -1.5    # v
      roll = -2           # v

      # common rewards
      ang_vel_xy = -0.2       # v
      dof_acc = -7.5e-7       # v
      collision = -10.        # v
      action_rate = -0.015    # v
      dof_pos_limits = -10.0  # v
      delta_torques = -1.0e-7 # v
      hip_pos = -0.3          # v
      work = -0.003           # v
      feet_jerk = -0.0002             # v
      feet_drag = -0.08               # v
      feet_contact_forces = -0.001    # v 最大接触力惩罚，*2s前无奖励* episode_length_buf
      base_height = -5.0          # v
