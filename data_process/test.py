import decord
vr = decord.VideoReader("/home/chenyipeng/data/real_robot_data_process/swap_three_cubes/videos/chunk-000/observation.images.image/episode_000000.mp4")
print(len(vr))