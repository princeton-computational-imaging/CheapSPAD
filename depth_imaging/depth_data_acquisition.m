% Supplemental code for SIGGRAPH 2021 paper "Low-Cost SPAD Sensing for Non-Line-Of-Sight Tracking, Material Classification and Depth Imaging"
% Author: Clara Callenberg

% Script for acquiring depth images with the VL53L1X and a galvo-mirror
% system operated with an Arduino. Adjust COM-ports for your system. 

%% Sensor Serial Port
s = serialport('COM5', 460800);

%% Arduino Serial Port
arduino = serialport('COM6', 115200);

%% Output File Name
filename = 'depth_image.mat';

%%
% scan size (must correspond to settings in Arduino code)
n_x = 128;
n_y = 128;
 
write(arduino, 'r', 'char'); % reset mirrors

data = '';

datacube = zeros(n_y, n_x, 24);

image = zeros(n_y, n_x);

bufferhistograms = 1;  % set this higher if you feel that there are problems with sync between mirrors and sensor (will slow down acquisition though)

figure('Position', [200, 200, 800, 800]);

imagesc(image);
axis square;
title('Scanning Scene');
drawnow;

i_measurement = 1;
n_measurements = n_x*n_y;
histcount = 0;
histo_smoothed = zeros(1,24);

% measurement loop
while(1)
    
    flush(s);
    data = readline(s);
    measOK = 1;
    
    if (strlength(data) > 5)
        histcount = histcount + 1;
        if histcount == bufferhistograms
            histo = [];
            while sum(size(histo) == [1, 25]) ~= 2
                try
                    histo = cell2mat(textscan(data, '%d[%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]'));
                catch
                    data = readline(s);
                end
            end
            histo = histo(2:end);
            histcount = 0;
            if sum(size(histo) == [1, 24]) ~= 2
                disp('missed measurement');
                histo = zeros(1, 24);
            end
            xpos = ceil(i_measurement/n_x);
            ypos = mod(i_measurement-1, n_y)+1;
            datacube(ypos, xpos, :) = histo;
            v = sum(double(histo(6:end)).*[6:24])/sum(histo(6:end));
            image(ypos, xpos) = v;
            imagesc(image, [6,24]);
             drawnow;
             if measOK
                 i_measurement = i_measurement + 1;
                 if i_measurement > n_measurements
                     break;
                 end
                 
                 write(arduino, 's', 'char');
                 s.flush()
             end
        end
        
        
    end
    
    
end

save(['./data/' filename], 'datacube');
disp([filename ' saved.']);




