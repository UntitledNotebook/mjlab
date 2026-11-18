.. _motion-imitation:

Motion Imitation
================

mjlab can train humanoid policies to imitate reference motions. This page
covers motion data preprocessing and training.

WandB registry setup
--------------------

mjlab uses `Weights & Biases <https://wandb.ai/>`_ to store and load
reference motions. Before preprocessing any motions, create a WandB registry
by following the
`BeyondMimic instructions <https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup>`_
(only the registry creation step; skip the ``csv_to_npz.py`` command shown
there).

Motion preprocessing
--------------------

Reference motions are retargeted CSV files in Unitree's generalized
coordinate convention (base position, base quaternion in xyzw, then joint
angles).

Convert a CSV to the NPZ format mjlab expects:

.. code-block:: bash

   MUJOCO_GL=egl uv run -m mjlab.scripts.csv_to_npz \
       --input-file <PATH_TO_CSV> \
       --output-name <MOTION_NAME> \
       --input-fps 30 \
       --output-fps 50 \
       --render True

The script plays the motion through MuJoCo Warp, computes forward kinematics
for every body, and uploads the resulting NPZ to your WandB registry.

.. warning::

   The BeyondMimic task (``Mjlab-Tracking-Flat-Unitree-G1``) requires
   mjlab's converter (``mjlab.scripts.csv_to_npz``).
   Converters from other frameworks such as IsaacLab produce NPZ files with
   incompatible body orderings. The NPZ stores precomputed body positions and
   quaternions indexed by body number, and different physics engines assign
   body indices differently (MuJoCo uses depth first traversal, PhysX uses
   breadth first). A mismatched NPZ will map tracking targets to the wrong
   bodies and training will not converge.

Training
--------

.. code-block:: bash

   uv run train Mjlab-Tracking-Flat-Unitree-G1 \
       --registry-name your-org/motions/motion-name \
       --env.scene.num-envs 4096

Evaluation
----------

.. code-block:: bash

   uv run play Mjlab-Tracking-Flat-Unitree-G1 \
       --wandb-run-path your-org/mjlab/run-id

.. _scaletrack:

ScaleTrack
----------

``Mjlab-ScaleTrack-Flat-Unitree-G1`` runs the ScaleBFM XL humanoid transformer
directly from its PyTorch checkpoint. It accepts a 50 Hz G1 motion NPZ produced
by ScaleTrack's packager or a ScaleTrack YAML mapping motion names to NPZ paths.
The task uses the joint and body order recorded by Isaac Lab. The actor updates
29 joint position targets at 50 Hz using three frames of state and six frames
of reference poses.

.. code-block:: bash

   S=/path/to/ScaleBFM/ScaleTrack
   uv run play Mjlab-ScaleTrack-Flat-Unitree-G1 \
       --checkpoint-file "$S/logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_xl/model_22200.pt" \
       --motion-file "$S/source/scaletrack/data/example/example.yaml" --num-envs 3

Playback starts at frame zero with nominal robot parameters. Each environment
starts on a different clip when enough clips are available and advances to the
next clip on completion. Whole-body tracking is the default. Select a subset
with ``--mode-index`` (0 pelvis, 1 hands, 2 pelvis and hands, 3 hands and feet,
4 pelvis, hands and feet, 5 upper body, 6 pelvis and upper body, 7 whole body).
Pelvis modes also support ``--local-tracking True``.

The native training entry point accepts the same motion files. It samples
tracking modes and uses mjlab's RSL-RL PPO optimizer. Importing a ScaleBFM
checkpoint for fine-tuning initializes the actor and critic with a fresh
optimizer.

A motion YAML can set a fixed sampling weight for each clip. In this example,
training selects ``weishuai-136`` on 75% of new draws on average.

.. code-block:: yaml

   weishuai-136: {path: weishuai-136.npz, weight: 3}
   weishuai-205: weishuai-205.npz

Paths can be relative to the YAML file. A bare path or an omitted ``weight``
has weight 1. Weights must be finite and nonnegative, with at least one positive
weight. A clip with weight 0 is available for playback and excluded from
training draws.

.. code-block:: bash

   uv run train Mjlab-ScaleTrack-Flat-Unitree-G1 \
       --env.commands.motion.motion-file "$S/source/scaletrack/data/example/example.yaml" \
       --env.scene.num-envs 64

To reproduce the XL playback check against the original Isaac Lab joint targets
and tracking errors on all three example clips, run

.. code-block:: bash

   SCALETRACK_ROOT="$S" uv run pytest -s tests/test_scaletrack.py -k xl_checkpoint
