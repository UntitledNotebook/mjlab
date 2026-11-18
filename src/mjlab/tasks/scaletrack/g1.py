"""G1 layout used by the ScaleTrack motion packager and policy."""

from dataclasses import replace

from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1
from mjlab.entity import EntityCfg

# Isaac Lab enumerates the articulation breadth first. The NPZs have no names.
JOINT_NAMES = (
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
)
BODY_NAMES = ("pelvis",) + tuple(
  "torso_link" if joint == "waist_pitch_joint" else joint.replace("_joint", "_link")
  for joint in JOINT_NAMES
)
TRACKED_BODIES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)

# Columns index TRACKED_BODIES in the eight published ScaleTrack modes.
TRACKING_MODES = (
  (0,),
  (10, 13),
  (0, 10, 13),
  (10, 13, 3, 6),
  (0, 10, 13, 3, 6),
  (8, 9, 10, 11, 12, 13),
  (0, 8, 9, 10, 11, 12, 13),
  tuple(range(14)),
)

ACTION_SCALE = {
  **g1.G1_ACTION_SCALE,
  "waist_yaw_joint": 0.25,
  "waist_roll_joint": 0.25,
  "waist_pitch_joint": 0.25,
}


def robot_cfg() -> EntityCfg:
  hips = replace(
    g1.G1_ACTUATOR_7520_14,
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
  )
  waist_yaw = replace(
    g1.G1_ACTUATOR_7520_14,
    target_names_expr=("waist_yaw_joint",),
    stiffness=100.0,
    damping=2.0,
  )
  waist = replace(g1.G1_ACTUATOR_WAIST, stiffness=300.0, damping=5.0)
  return replace(
    g1.get_g1_robot_cfg(),
    articulation=replace(
      g1.G1_ARTICULATION,
      soft_joint_pos_limit_factor=0.95,
      actuators=(
        g1.G1_ACTUATOR_5020,
        hips,
        g1.G1_ACTUATOR_7520_22,
        g1.G1_ACTUATOR_4010,
        waist,
        waist_yaw,
        g1.G1_ACTUATOR_ANKLE,
      ),
    ),
    collisions=(replace(g1.FULL_COLLISION, friction=(1.0,)),),
  )
