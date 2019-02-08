add_executable(ibr_render ibr_render.cc)
target_link_libraries(ibr_render replay)

add_executable(min_composite min_composite.cc)
target_link_libraries(min_composite replay)

add_executable(test_render test_render.cc)
target_link_libraries(test_render replay)

#add_executable(render_vr180_scene render_vr180_scene.cc)
#target_link_libraries(render_vr180_scene replay)

#add_executable(render_vr180_video render_vr180_video.cc)
#target_link_libraries(render_vr180_video replay)

#add_executable(dump_vr180_frames dump_vr180_frames.cc)
#target_link_libraries(dump_vr180_frames replay)

#add_executable(reconstruct_vr180_pano reconstruct_vr180_pano.cc)
#target_link_libraries(reconstruct_vr180_pano replay)
