#!/usr/bin/env python
import os
import shutil
import time 

process_all = False

# Remove the stale files
# os.system("rm -f output/*.mp4 && rm -f output/*.png")

dataset_folder = '/Users/holynski/Drive/research/datasets/windows/'

datasets = os.listdir(dataset_folder)
datasets.sort()

for dataset in datasets:
    if (dataset == 'belgrave'):
        continue
    if (dataset == 'gold_parking1'):
        continue
    if (dataset == 'blinds'):
        continue
    if (dataset == 'closeup'):
        continue
    if (dataset == 'courthouse'):
        continue
    if (dataset == 'elwha2'):
        continue
    # if (dataset == 'promotion'):
        # continue
    dataset_path = os.path.join(dataset_folder, dataset)
    if not os.path.exists(os.path.join(dataset_path,'textured.ply')):
        continue
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>--------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("                               Processing " + dataset)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>--------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")


    if (os.system("make -j") != 0):
        quit()

    result_time = time.strftime("%y_%m_%d_%H_%M_%S", time.gmtime()) 
    results_folder = os.path.join('/Users/holynski/results/'+dataset, result_time)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    bundler_args = ' --bundler_file='+os.path.join(dataset_path, 'bundler.out') + \
            ' --image_list='+os.path.join(dataset_path, 'frames_undistorted/list.lst')+\
            ' --images_directory='+os.path.join(dataset_path, 'frames_undistorted/')+\
            ' --output_reconstruction='+os.path.join(dataset_path, 'reconstruction.bin')+\
            ' --exposure_mesh='+os.path.join(dataset_path, 'textured.ply')+\
            ' --exposure_align=1'+\
            ' --logtostderr'
    if not os.path.exists(os.path.join(dataset_path, 'reconstruction.bin')):
        print("Converting bundler file!")
        if (os.system('./bin/convert_bundler ' + bundler_args) != 0):
            print("Failed to convert bundler file!")
            quit()

    args = ' --reconstruction='+os.path.join(dataset_path, 'reconstruction.bin') + \
           ' --images_directory='+os.path.join(dataset_path, 'frames_undistorted/')+\
           ' --output_directory='+results_folder+\
           ' --mesh='+os.path.join(dataset_path, 'textured.ply')+\
           ' --window_mesh='+os.path.join(dataset_path, 'windows.ply')+\
           ' --logtostderr'+\
           ' --v=2'

    if (os.system('./bin/separate_reflections_mrf ' + args) != 0):
        # shutil.rmtree(results_folder)
        quit()
    
    if not os.path.exists(os.path.join(dataset_path,'min_composite')):
        os.makedirs(os.path.join(dataset_path,'min_composite'))
    os.system('ffmpeg -framerate 30 -pattern_type glob -i \'' + os.path.join(results_folder,'reprojected_*.png') + '\' -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p ' + os.path.join(results_folder, 'reprojected.mp4')) 
    os.system('ffmpeg -framerate 30 -pattern_type glob -i \'' + os.path.join(results_folder,'residual_*.png') + '\' -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p ' + os.path.join(results_folder, 'residual.mp4')) 
    os.system('ffmpeg -framerate 30 -pattern_type glob -i \'' + os.path.join(results_folder,'composed_*.png') + '\' -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p ' + os.path.join(results_folder, 'composed.mp4')) 
    os.system('ffmpeg -framerate 30 -pattern_type glob -i \'' + os.path.join(results_folder,'aligned_residual_*.png') + '\' -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p ' + os.path.join(results_folder, 'aligned_residual.mp4')) 
    os.system('ffmpeg -framerate 30 -pattern_type glob -i \'' + os.path.join(results_folder,'per_view_residual_*.png') + '\' -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p ' + os.path.join(results_folder, 'per_view_residual.mp4')) 
    os.system('ffmpeg -framerate 30 -pattern_type glob -i \'' + os.path.join(results_folder,'second_plane_*.png') + '\' -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p ' + os.path.join(results_folder, 'aligned_second_layer.mp4')) 
    os.system('ffmpeg -framerate 30 -pattern_type glob -i \'' + os.path.join(results_folder,'flow_*.png') + '\' -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p ' + os.path.join(results_folder, 'flow_aligned.mp4')) 
