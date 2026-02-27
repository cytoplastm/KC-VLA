#for env_id in PickCube-v1 PushCube-v1 StackCube-v1 PullCube-v1 PullCubeTool-v1 DrawTriangle-v1 PlaceSphere-v1 LiftPegUpright-v1 PegInsertionSide-v1
for env_id in PullCube-v1
do
    python  mani_skill/examples/motionplanning/xarm6/run.py \
        --env-id $env_id \
        --traj-name="trajectory_xarm6" \
        -n 300 \
        --only-count-success \
        --num-procs 8 \
        # --sim-backend="gpu"
done