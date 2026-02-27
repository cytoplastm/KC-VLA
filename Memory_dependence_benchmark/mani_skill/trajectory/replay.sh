# for robot in panda xarm6
#   do
#     for env_id in PickCube-v1 PushCube-v1 StackCube-v1 PullCube-v1 PullCubeTool-v1 PlaceSphere-v1 LiftPegUpright-v1 PegInsertionSide-v1 DrawTriangle-v1 DrawSVG-v1
#       do
#           python mani_skill/trajectory/replay_trajectory.py \
#               --traj_path="/home/wangzhibin/ManiSkill-main/data/${robot}/${env_id}/motionplanning/trajectory_${robot}.h5" \
#               -o rgb \
#               -c pd_ee_delta_pose \
#               --save_traj \
#               --num-envs 10 \
#               -b physx_cpu
#       done
#   done

for robot in widowxai
  do
    for env_id in PickCube-v1 PushCube-v1 StackCube-v1 PullCube-v1 PullCubeTool-v1 PlaceSphere-v1 LiftPegUpright-v1
      do
          python /home/pc/ManiSkill/mani_skill/trajectory/replay_trajectory.py \
              --traj_path="/home/pc/maniskill/data/${robot}/${env_id}/motionplanning/trajectory_${robot}.h5" \
              -o rgb \
              -c pd_ee_delta_pose \
              --save_traj \
              --num-envs 10 \
              -b physx_cpu
      done
  done