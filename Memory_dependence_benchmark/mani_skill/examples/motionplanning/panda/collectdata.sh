for env_id in TeacherArmShuffle-v1 SwapThreeCubes-v1 ickPlaceThreetimes-v1 PushCubeWithSignal-v1
do
    python  mani_skill/examples/motionplanning/panda/run.py \
        --env-id $env_id \
        --traj-name="trajectory_panda" \
        -n 100 \
        --only-count-success \
        --num-procs 1 \
        --save-video \
        # --use-env-states
        # --sim-backend="gpu"
done

# bash mani_skill/examples/motionplanning/panda/collectdata.sh