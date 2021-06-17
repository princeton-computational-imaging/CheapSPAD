% Supplemental code for SIGGRAPH 2021 paper "Low-Cost SPAD Sensing for Non-Line-Of-Sight Tracking, Material Classification and Depth Imaging"
% Author: Clara Callenberg

% Script that reads in the raw NLOS measurments and combines them into a
% file for further processing. Also calculates distances in the setup scene
% (need to be adjusted accordingly). 

outfilename = 'NLOS_measurement.mat';   % name of the output file

files = dir('*.mat');
n_meas = size(files, 1);
datacubes = {};
positions = {}; 

f = waitbar(0, 'Loading measurement data');
for i = 1:n_meas
    waitbar((i-1)/n_meas, f);
    load(sprintf('./meas%d.mat', i));
    datacubes{i} = datacube;
    positions{i} = position;
end
close(f);


% position of the VL53L1X
spadpos = [-0.83, -1.06, 0.95]'; 

% positions of "probe points" on wall
tl = [1.06-1.67, 0, 1.53];
bl = [1.06-1.60, 0, 1.08];
tr = [1.06-0.52, 0, 1.99];
br = [1.06-0.57, 0, 1.21];


cornerpos = zeros(2,2,3);
cornerpos(1,1,:) = tl;
cornerpos(1,2,:) = tr;
cornerpos(2,1,:) = bl;
cornerpos(2,2,:) = br;

d_sw = zeros(2,2);
d_sw(1,1) = norm(squeeze(cornerpos(1,1,:)) - spadpos);
d_sw(1,2) = norm(squeeze(cornerpos(1,2,:)) - spadpos);
d_sw(2,1) = norm(squeeze(cornerpos(2,1,:)) - spadpos);
d_sw(2,2) = norm(squeeze(cornerpos(2,2,:)) - spadpos);

dists_spad_wall = {};
dists_wall_obj = {};
for i = 1:n_meas
    pos = positions{i};
    
    dists_spad_wall{i} = d_sw;
    
    d_wo = zeros(2,2);
    
    d_wo(1,1) = norm(squeeze(cornerpos(1,1,:)) - pos);
    d_wo(1,2) = norm(squeeze(cornerpos(1,2,:)) - pos);
    d_wo(2,1) = norm(squeeze(cornerpos(2,1,:)) - pos);
    d_wo(2,2) = norm(squeeze(cornerpos(2,2,:)) - pos);
   
    dists_wall_obj{i} = d_wo;
    
end

save(outfilename, 'datacubes', 'positions', 'dists_spad_wall', 'dists_wall_obj');