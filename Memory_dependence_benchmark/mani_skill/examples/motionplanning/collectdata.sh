# for robot in panda xarm6
#   do
#     for env_id in PickCube-v1 PushCube-v1 StackCube-v1 PullCube-v1 PullCubeTool-v1 PlaceSphere-v1 LiftPegUpright-v1 PegInsertionSide-v1 DrawTriangle-v1 DrawSVG-v1
#     do
#         python mani_skill/examples/motionplanning/$robot/run.py \
#             --env-id $env_id \
#             --traj-name="trajectory_${robot}" \
#             -n 150 \
#             --only-count-success \
#             --num-procs 8
#             # --sim-backend="gpu"
#     done
#   done

for robot in xarm7
    do
        # for env_id in PickCube-v1 PushCube-v1 StackCube-v1 PullCube-v1 PullCubeTool-v1 PlaceSphere-v1 LiftPegUpright-v1
        for env_id in PushCube-v1
            do
                python /home/pc/ManiSkill/mani_skill/examples/motionplanning/$robot/run.py \
                    --env-id $env_id \
                    --traj-name="trajectory_${robot}" \
                    -n 200 \
                    --num-procs 8 \
                    --only-count-success \
                    # --sim-backend="gpu"
                    # --save-video \
    done
done

for robot in widowxai
    do
        # for env_id in PickCube-v1 PushCube-v1 StackCube-v1 PullCube-v1 PullCubeTool-v1 PlaceSphere-v1 LiftPegUpright-v1
        for env_id in PickCube-v1 
            do
                python /home/pc/ManiSkill/mani_skill/examples/motionplanning/$robot/run.py \
                    --env-id $env_id \
                    --traj-name="trajectory_${robot}" \
                    -n 200 \
                    --only-count-success \
                    --num-procs 8
                    # --sim-backend="gpu"
    done
done